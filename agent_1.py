import os
import json
import subprocess
import sys
import threading
from queue import Queue, Empty, Full
from typing import Annotated, TypedDict, List, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

# --- Environment Setup ---
# Load environment variables from .env file for API keys
load_dotenv()

# Ensure you have GROQ_API_KEY, GOOGLE_API_KEY, GOOGLE_CX, and WEATHER_API_KEY in your .env file

# --- MCP Client for Tool Execution ---

class MCPClient:
    """
    A client to communicate with the MCP server running as a subprocess.
    It sends JSON-RPC requests to the server's stdin and reads responses from its stdout.
    """
    def __init__(self, server_script_path):
        self.server_script_path = server_script_path
        self.process = None
        self.request_id = 0
        self.pending_requests = {}
        self.lock = threading.Lock()

    def start(self):
        """Starts the MCP server subprocess."""
        self.process = subprocess.Popen(
            [sys.executable, self.server_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run server from its directory
        )
        # Start a thread to read stdout and stderr from the server
        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._read_error, daemon=True).start()
        print("MCP server process started.")

    def _read_output(self):
        """Reads stdout from the server and puts responses into a queue."""
        for line in iter(self.process.stdout.readline, ''):
            try:
                response = json.loads(line)
                req_id = response.get("id")
                with self.lock:
                    if req_id in self.pending_requests:
                        self.pending_requests[req_id].put(response)
            except json.JSONDecodeError:
                print(f"Agent: Received non-JSON output from server: {line.strip()}", file=sys.stderr)

    def _read_error(self):
        """Reads and prints stderr from the server for debugging."""
        for line in iter(self.process.stderr.readline, ''):
            print(f"Server STDERR: {line.strip()}", file=sys.stderr)

    def stop(self):
        """Stops the MCP server subprocess."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("MCP server process stopped.")

    def _send_request(self, method, params):
        """Sends a JSON-RPC request to the server."""
        response_queue = Queue()
        with self.lock:
            self.request_id += 1
            req_id = self.request_id
            self.pending_requests[req_id] = response_queue
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": req_id,
            }
            if self.process and self.process.stdin:
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
            return req_id

    def _get_response(self, req_id):
        """Retrieves a response for a given request ID from the queue."""
        with self.lock:
            q = self.pending_requests.get(req_id)
        if not q:
            return "Error: Request ID not found."
        try:
            response = q.get(timeout=30)
            if "result" in response:
                return response["result"]
            elif "error" in response:
                return f"Error from server: {response['error'].get('message', 'Unknown error')}"
        except Empty:
            return "Error: No response from server (timeout)."
        finally:
            with self.lock:
                self.pending_requests.pop(req_id, None)

    def list_tools(self):
        """Lists tools available on the MCP server."""
        req_id = self._send_request("tools/list", {})
        return self._get_response(req_id)

    def call(self, tool_name: str, *args, **kwargs):
        """Makes a tool call to the MCP server."""
        # The MCP server expects params to be a dict for keyword arguments
        # or a list for positional arguments.
        if kwargs:
            arguments = kwargs
        elif args:
            arguments = list(args)
        else:
            arguments = {}
        
        params = {"name": tool_name, "arguments": arguments}
        req_id = self._send_request("tools/call", params)
        result = self._get_response(req_id)

        # The MCP server wraps successful results. We need to unwrap them
        # to get the actual tool output string for the agent.
        if (isinstance(result, dict) and 
            'content' in result and 
            isinstance(result['content'], list) and 
            len(result['content']) > 0 and
            'text' in result['content'][0]):
            return result['content'][0]['text']

        return result

    def initialize(self):
        """Sends an initialize request to the server to start a session."""
        params = {
            "protocolVersion": "1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "LangGraphAgent",
                "version": "0.1"
            }
        }
        req_id = self._send_request("initialize", params)
        response = self._get_response(req_id)
        if isinstance(response, str) and "Error" in response:
            print(f"Agent: Failed to initialize MCP server: {response}", file=sys.stderr)
            return False
        print("Agent: MCP server initialized successfully.")
        return True

# --- Agent Setup ---

# Initialize the MCP client
mcp_client = MCPClient("server.py")

import inspect

def create_langchain_tool(mcp_client, tool_spec):
    """Dynamically creates a LangChain tool from an MCP tool specification."""
    tool_name = tool_spec['name']
    tool_description = tool_spec.get('description', '')
    
    # Extract parameters from inputSchema
    input_schema = tool_spec.get('inputSchema', {})
    properties = input_schema.get('properties', {})
    required = input_schema.get('required', [])

    type_map = {
        'string': str,
        'integer': int,
        'number': float,
        'boolean': bool,
        'array': list,
        'object': dict,
    }

    param_strings = []
    args_dict_items = []
    
    for param_name, param_info in properties.items():
        json_type = param_info.get('type', 'string')
        py_type = type_map.get(json_type, Any)
        
        # Use the type name for the signature
        if hasattr(py_type, '__name__'):
            type_name = py_type.__name__
        else:
            type_name = 'Any'

        if param_name in required:
            param_strings.append(f"{param_name}: {type_name}")
        else:
            param_strings.append(f"{param_name}: {type_name} = None")
        
        args_dict_items.append(f"'{param_name}': {param_name}")

    signature_str = ", ".join(param_strings)
    args_dict_str = ", ".join(args_dict_items)
    
    # Create the function source code
    func_code = f"""
def {tool_name}({signature_str}):
    '''{tool_description}'''
    args = {{{args_dict_str}}}
    # Remove None values to allow server defaults to kick in
    args = {{k: v for k, v in args.items() if v is not None}}
    return mcp_client.call('{tool_name}', **args)
"""
    
    # Execute the code to define the function
    local_scope = {}
    exec(func_code, {'mcp_client': mcp_client, 'Any': Any, **type_map}, local_scope) # Pass type_map as globals
    tool_func = local_scope[tool_name]

    return tool(tool_func)

def discover_tools(mcp_client):
    """Discovers tools from the MCP server and creates LangChain tools."""
    response = mcp_client.list_tools()
    try:
        if isinstance(response, dict) and 'tools' in response:
            return [create_langchain_tool(mcp_client, spec) for spec in response['tools']]
    except Exception as e:
        print(f"Error parsing tools: {e}", file=sys.stderr)
    
    print(f"Error: 'list_tools' did not return expected format. Got: {response}", file=sys.stderr)
    return []

# --- LangGraph State and Nodes ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# Use Groq as the LLM. Ensure GROQ_API_KEY is in your .env file.
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
model_with_tools = None # Will be initialized after tool discovery

# Define the prompt template with a system message to guide the AI's behavior
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the provided tools to answer the user's questions. "
                   "Prioritize using specific tools (like get_weather and math tools) over the internet_search tool. "
                   "Only use internet_search if the specific tools cannot answer the query or if the query requires general knowledge. "
                   "If the user asks to perform a specific action (like booking a flight, ordering food, etc.) that you cannot do with the available tools, "
                   "do not use any tools. Instead, reply exactly: 'As of now I can't do it but I will learn for you'. "
                   "Only respond to the user's most recent query, using the conversation history for context if necessary."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def should_continue(state: AgentState) -> str:
    """Determines whether to continue with another tool call or end."""
    last_message = state['messages'][-1]
    # If the model decided to call a tool, route to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, the model provided a final answer, so we can end.
    return END

def call_model(state: AgentState) -> dict:
    """The primary node that calls the LLM to decide the next action."""
    chain = prompt | model_with_tools
    response = chain.invoke({"messages": state['messages']})
    return {"messages": [response]}

# --- Graph Definition ---

workflow = StateGraph(AgentState)

# --- Main Interaction Loop ---

if __name__ == "__main__":
    mcp_client.start()
    try:
        # Initialize the connection with the server
        if not mcp_client.initialize():
            raise RuntimeError("Could not initialize connection with MCP server.")

        # --- Dynamic Tool Discovery and Graph Re-compilation ---
        print("Discovering tools from MCP server...")
        tools = discover_tools(mcp_client)
        model_with_tools = llm.bind_tools(tools) # Bind tools to the LLM

        # Define the graph structure AFTER tools are discovered
        tool_node = ToolNode(tools)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        app = workflow.compile()
        print(f"Discovered and configured {len(tools)} tools.")
        messages = []
        print("Agent is ready. How can I help you today? (type 'exit' to quit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            messages.append(HumanMessage(content=user_input))
            
            result = app.invoke({"messages": messages})
            
            # Update the conversation history with the full result from the graph
            messages = result['messages']

            # Extract the actual text content for printing, handling Gemini's specific format
            ai_content = messages[-1].content
            if isinstance(ai_content, list):
                # Handle Gemini's list-of-dicts format
                text_parts = [part['text'] for part in ai_content if isinstance(part, dict) and 'text' in part]
                display_text = "\n".join(text_parts)
            else:
                # Handle standard string content
                display_text = ai_content
            print(f"AI: {display_text}")
    finally:
        mcp_client.stop()
        print("Agent has shut down.")
