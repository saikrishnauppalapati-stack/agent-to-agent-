import os
import json
import subprocess
import sys
import threading
import httpx
from queue import Queue, Empty
from typing import Annotated, TypedDict, List, Any
from pydantic import create_model, Field

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

# --- Environment Setup ---
# Load environment variables from .env file for API keys
load_dotenv()

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
                if line.strip():
                    print(f"Agent: Received non-JSON output from server: {line.strip()}", file=sys.stderr)

    def _read_error(self):
        """Reads and prints stderr from the server for debugging."""
        for line in iter(self.process.stderr.readline, ''):
            # Suppress stderr output to keep the console clean for the user
            pass

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

    def _send_notification(self, method, params):
        """Sends a JSON-RPC notification (no ID) to the server."""
        with self.lock:
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params
            }
            if self.process and self.process.stdin:
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()

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
        if kwargs:
            arguments = kwargs
        elif args:
            arguments = list(args)
        else:
            arguments = {}
        
        params = {"name": tool_name, "arguments": arguments}
        req_id = self._send_request("tools/call", params)
        result = self._get_response(req_id)

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
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "LangGraphAgent", "version": "0.1"}
        }
        req_id = self._send_request("initialize", params)
        response = self._get_response(req_id)
        if isinstance(response, str) and "Error" in response:
            print(f"Agent: Failed to initialize MCP server: {response}", file=sys.stderr)
            return False
        
        # Send initialized notification
        self._send_notification("notifications/initialized", {})
        print("Agent: MCP server initialized successfully.")
        return True

# --- Agent Setup ---

# Initialize the MCP client pointing to your specific server file
mcp_client = MCPClient("mcp_kb_server.py")

def create_langchain_tool(mcp_client, tool_spec):
    """Dynamically creates a LangChain tool from an MCP tool specification."""
    tool_name = tool_spec['name']
    tool_description = tool_spec.get('description', '')
    input_schema = tool_spec.get('inputSchema', {})
    properties = input_schema.get('properties', {})
    required = input_schema.get('required', [])
    
    # Create Pydantic model for arguments
    fields = {}
    for prop_name, prop_def in properties.items():
        prop_type = Any
        if prop_def.get('type') == 'string':
            prop_type = str
        elif prop_def.get('type') == 'integer':
            prop_type = int
        elif prop_def.get('type') == 'number':
            prop_type = float
        elif prop_def.get('type') == 'boolean':
            prop_type = bool
        
        description = prop_def.get('description', '')
        
        if prop_name in required:
            fields[prop_name] = (prop_type, Field(description=description))
        else:
            fields[prop_name] = (prop_type | None, Field(default=None, description=description))
            
    ArgsModel = create_model(f"{tool_name}Arguments", **fields)
    ArgsModel.__doc__ = tool_description
    
    # Create a dynamic function that calls the MCP client
    def tool_func(**kwargs):
        return mcp_client.call(tool_name, **kwargs)
    
    tool_func.__name__ = tool_name
    tool_func.__doc__ = tool_description
    
    return tool(args_schema=ArgsModel)(tool_func)

def discover_tools(mcp_client):
    """Discovers tools from the MCP server and creates LangChain tools."""
    response = mcp_client.list_tools()
    if isinstance(response, dict) and 'tools' in response:
        return [create_langchain_tool(mcp_client, spec) for spec in response['tools']]
    return []

# --- LangGraph State and Nodes ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

llm = ChatGroq(model_name="llama-3.1-8b-instant")

workflow = StateGraph(AgentState)

memory = MemorySaver()

def call_model(state: AgentState):
    messages = state['messages']
    # Inject a system message to guide the model's behavior
    if not messages or not isinstance(messages[0], SystemMessage):
        system_prompt = (
            "You are a helpful assistant connected to a strict knowledge base. "
            "You MUST use the provided tools to answer questions. "
            "Do NOT use your own internal knowledge or training data. "
            "If the information is not present in the tool output, state clearly that it is not in the database. "
            "Do NOT make up facts. "
            "Ensure tool calls are valid JSON."
        )
        messages = [SystemMessage(content=system_prompt)] + messages
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

if __name__ == "__main__":
    mcp_client.start()
    try:
        if mcp_client.initialize():
            print("Discovering tools...")
            tools = discover_tools(mcp_client)
            model_with_tools = llm.bind_tools(tools)
            
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", ToolNode(tools))
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges("agent", should_continue)
            workflow.add_edge("tools", "agent")
            app = workflow.compile(checkpointer=memory)
            
            print(f"Agent Ready! ({len(tools)} tools loaded)")
            print("Ask a question (or type 'exit'):")
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']: break
                
                try:
                    # Use a thread_id to maintain conversation history
                    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 25}
                    # Set recursion_limit to prevent infinite loops and rate limit issues
                    for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                        for value in event.values():
                            message = value["messages"][-1]
                            # Only print AI messages that have actual content (skipping tool outputs and tool calls)
                            if isinstance(message, AIMessage) and message.content:
                                print("Agent:", message.content)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print("Agent Error: Rate limit exceeded. Please wait a moment before trying again.")
                    else:
                        print(f"Agent Error: {e}")
                except Exception as e:
                    print(f"Agent Error: {e}")
    finally:
        mcp_client.stop()