import os
import json
import subprocess
import sys
import threading
from queue import Queue, Empty
from typing import Annotated, TypedDict, List, Any, Literal, Union
from pydantic import create_model, Field

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

# --- Environment Setup ---
load_dotenv()

# --- MCP Client for Tool Execution ---

class MCPClient:
    """
    A client to communicate with an MCP server running as a subprocess.
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
            encoding='utf-8',
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._read_error, daemon=True).start()

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
                pass

    def _read_error(self):
        """Reads stderr from the server."""
        for line in iter(self.process.stderr.readline, ''):
            # Keep stderr clean unless debugging is needed
            pass

    def stop(self):
        """Stops the MCP server subprocess."""
        if self.process:
            self.process.terminate()
            self.process.wait()

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
        req_id = self._send_request("tools/list", {})
        return self._get_response(req_id)

    def call(self, tool_name: str, *args, **kwargs):
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
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "Agent3_Orchestrator", "version": "0.1"}
        }
        req_id = self._send_request("initialize", params)
        response = self._get_response(req_id)
        if isinstance(response, str) and "Error" in response:
            # Fallback for older servers or different protocol versions
            print(f"Warning: Init failed ({response}), trying simple init...")
        return True

def create_langchain_tool(mcp_client, tool_spec):
    tool_name = tool_spec['name']
    tool_description = tool_spec.get('description', '')
    input_schema = tool_spec.get('inputSchema', {})
    properties = input_schema.get('properties', {})
    required = input_schema.get('required', [])
    
    fields = {}
    for prop_name, prop_def in properties.items():
        prop_type = Any
        t = prop_def.get('type')
        if t == 'string': prop_type = str
        elif t == 'integer': prop_type = int
        elif t == 'number': prop_type = float
        elif t == 'boolean': prop_type = bool
        
        if prop_name in required:
            fields[prop_name] = (prop_type, Field(description=prop_def.get('description', '')))
        else:
            fields[prop_name] = (prop_type | None, Field(default=None, description=prop_def.get('description', '')))
            
    ArgsModel = create_model(f"{tool_name}Arguments", **fields)
    
    def tool_func(**kwargs):
        return mcp_client.call(tool_name, **kwargs)
    
    tool_func.__name__ = tool_name
    tool_func.__doc__ = tool_description
    return tool(args_schema=ArgsModel)(tool_func)

# --- Agent Logic ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    plan_approved: bool

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

def planner_node(state: AgentState):
    """Analyzes the request and proposes a plan."""
    messages = state['messages']
    
    system_prompt = (
        "You are an Orchestrator Agent connecting two sub-agents:\n"
        "1. Agent 1 (General): Weather, Time, Math, Internet Search.\n"
        "2. Agent 2 (Knowledge Base): Company Policy, RAG.\n\n"
        "Your goal is to PLAN. Do NOT execute tools yet.\n"
        "Analyze the user's query. If tools are needed, output a plan in this EXACT format:\n"
        "'Plan: Agent [1 or 2] is used for this query to [reason]. I will call [Tool Name]. Give me permission to execute.'\n"
        "If no tools are needed, just answer the user."
    )
    
    # We use a fresh LLM call without tools bound to ensure it just plans
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [response]}

def human_approval_node(state: AgentState):
    """Pauses for human approval."""
    last_message = state['messages'][-1].content
    
    # Check if the agent actually proposed a plan
    if "Plan:" in last_message:
        print(f"\nAI Proposed Plan:\n{last_message}\n")
        user_input = input("Do you approve this plan? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y', 'ok']:
            return {"messages": [HumanMessage(content="Plan approved. Proceed to execute the tools.")], "plan_approved": True}
        else:
            return {"messages": [HumanMessage(content="Plan rejected. Stop.")], "plan_approved": False}
    else:
        # If no plan was proposed (just a chat response), we mark as approved to skip execution or just end
        return {"plan_approved": False}

def executor_node(state: AgentState):
    """Executes the tools if approved."""
    if not state.get("plan_approved"):
        return {}
    
    # Bind tools only for this node
    agent_with_tools = llm.bind_tools(all_tools)
    
    system_msg = SystemMessage(content=(
        "You are a helpful assistant. The user has approved your plan. "
        "You must now call the tools exactly as planned to answer the user's request. "
        "Do not ask for permission again. Just execute the tools. "
        "Ensure you use the correct tool name and arguments in JSON format."
    ))
    
    # Use the full conversation history so the model has context (Plan + Approval)
    response = agent_with_tools.invoke([system_msg] + state['messages'])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

# --- Main Execution ---

if __name__ == "__main__":
    # Initialize Clients
    client1 = MCPClient("agent_1_server.py")
    client2 = MCPClient("agent_2_server.py")
    
    client1.start()
    client2.start()
    
    try:
        print("Initializing agents...")
        client1.initialize()
        client2.initialize()
        
        # Discover Tools
        resp1 = client1.list_tools()
        if isinstance(resp1, dict):
            tools1 = [create_langchain_tool(client1, t) for t in resp1.get('tools', [])]
        else:
            print(f"Error listing tools from Agent 1: {resp1}")
            tools1 = []

        resp2 = client2.list_tools()
        if isinstance(resp2, dict):
            tools2 = [create_langchain_tool(client2, t) for t in resp2.get('tools', [])]
        else:
            print(f"Error listing tools from Agent 2: {resp2}")
            tools2 = []
            
        all_tools = tools1 + tools2
        print(f"Connected. Loaded {len(all_tools)} tools from Agent 1 & Agent 2.")

        # Build Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", planner_node)
        workflow.add_node("human_approval", human_approval_node)
        workflow.add_node("executor", executor_node)
        workflow.add_node("tools", ToolNode(all_tools))

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "human_approval")
        workflow.add_edge("human_approval", "executor")
        workflow.add_conditional_edges("executor", should_continue)
        workflow.add_edge("tools", "executor")

        app = workflow.compile()

        print("\nAgent 3 (Orchestrator) Ready. Type 'exit' to quit.")
        
        while True:
            user_query = input("\nYou: ")
            if user_query.lower() in ['exit', 'quit']:
                break
            
            # Run the graph
            # We initialize plan_approved to False for each new turn
            initial_state = {"messages": [HumanMessage(content=user_query)], "plan_approved": False}
            
            try:
                for event in app.stream(initial_state):
                    for key, value in event.items():
                        if key == "executor" and value and "messages" in value:
                            msg = value["messages"][-1]
                            if not msg.tool_calls: # Final answer
                                print(f"Agent: {msg.content}")
                        elif key == "planner":
                            # We don't print here, the human node handles the print/input
                            pass
            except Exception as e:
                print(f"Agent Error: {e}")

    finally:
        client1.stop()
        client2.stop()
        print("Agents stopped.")