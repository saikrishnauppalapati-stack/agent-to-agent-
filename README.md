# 🤖 Multi-Agent System using MCP, LangGraph & Groq LLM

This repository contains a multi-agent architecture built using **Model Context Protocol (MCP)**, **LangGraph**, and **Groq LLM**.  
The system supports **dynamic tool discovery**, **agent-to-agent orchestration**, and **human-in-the-loop execution control**, without hardcoding tools inside agents.

---

## 🧠 High-Level Architecture

┌──────────┐
│ User 👤 │
└────┬─────┘
↓
┌──────────────────────────────┐
│ Orchestrator Agent (Agent 3) │
│ • Planning │
│ • Human Approval │
└────┬─────────────────────────┘
↓
┌──────────────────────────────┐
│ MCP Clients 🔌 │
└────┬───────────────┬─────────┘
↓ ↓
┌─────────────┐ ┌─────────────────┐
│ Agent 1 │ │ Agent 2 │
│ General │ │ Knowledge (RAG) │
└────┬────────┘ └────┬────────────┘
↓ ↓
┌─────────────┐ ┌─────────────────┐
│ MCP Server │ │ MCP KB Server │
│ Tools 🧰 │ │ Vector Search 📚│
└────┬────────┘ └────┬────────────┘
↓ ↓
🌐 External APIs 🔎 FAISS Vector DB

yaml
Copy code

---

## ✨ Core Design Principles

- 🔄 Dynamic tool discovery at runtime  
- 🧩 Strict separation of reasoning and execution  
- ✋ Human-in-the-loop safety controls  
- 🔌 Protocol-driven agent communication  
- 📈 Modular and extensible architecture  

---

## 🛠️ Agent 1 – General Tools Agent

### Purpose
Handles real-world utility and computation tasks.

### Files
- `agent_1.py`
- `agent_1_server.py`

### Capabilities
- 🌦️ Weather retrieval  
- 🌍 Internet search  
- ➕➖✖️➗ Math operations (add, subtract, multiply, divide, sqrt)

### Internal Flow

User Query
↓
LLM decides tool usage
↓
tools/list (MCP)
↓
LangChain tool conversion
↓
Tool execution via MCP server

yaml
Copy code

---

## 📚 Agent 2 – Knowledge Base Agent (RAG)

### Purpose
Answers queries using internal documents only.

### Files
- `agent_2.py`
- `agent_2_server.py`

### Capabilities
- 🔍 Semantic search  
- 📄 List documents  
- 📖 Read full document content  

---

### 🧠 RAG Pipeline

Raw Documents
↓
Chunking ✂️
↓
Embeddings (HuggingFace)
↓
FAISS Vector Store
↓
Query Embedding
↓
Similarity Search 🔎
↓
Top-K Relevant Chunks

yaml
Copy code

### Constraint
The LLM must answer **only from tool output**.  
If information is not found, the agent explicitly reports unavailability.

---

## 🧭 Agent 3 – Orchestrator (Planner + Human Approval)

### Purpose
Coordinates agents, plans execution, and enforces human approval.

### File
- `agent_3.py`

---

### 🔁 Execution Lifecycle

User Query
↓
Planner Node 🧠 (No tools)
↓
Execution Plan 📋
↓
Human Approval ✋
↓
Executor Node ⚙️ (Tools enabled)
↓
Final Response ✅

yaml
Copy code

---

### 📋 Example Plan

Plan:
Agent 2 will be used to retrieve internal policy information.
Tool: search_knowledge_base
Requesting permission to execute.

yaml
Copy code

---

## 🔌 Model Context Protocol (MCP)

MCP is the communication layer between agents and tools.

### Characteristics
- 🔐 Tool isolation via subprocesses  
- 📜 JSON-RPC messaging  
- 🔍 Runtime tool discovery  
- 🧩 Language-agnostic protocol  

### Transport
- STDIO (stdin / stdout)

---

## 🔄 Dynamic Tool Discovery Flow

Start MCP Server
↓
tools/list
↓
Parse tool schemas
↓
Convert to LangChain tools
↓
Bind tools to LLM at runtime

yaml
Copy code

---

## 🕸️ LangGraph Execution Model

LangGraph defines stateful, deterministic agent workflows.

### Responsibilities
- 🧠 Agent state management  
- 🔁 LLM ↔ Tool looping  
- ⏸️ Pause and resume execution  
- ✋ Human-in-the-loop gating  
- 🔀 Conditional routing  

### Core Nodes
- Planner  
- Human Approval  
- Executor  
- Tools  

---

## ⚙️ Environment & Dependencies

### Python Requirements

python-dotenv
pydantic
requests
httpx
mcp
langchain-core
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
langgraph
faiss-cpu
sentence-transformers

yaml
Copy code

Environment variables are loaded using a `.env` file.

---

## 📌 Current Project Status

- ✅ All agents operate independently  
- 🔗 Orchestrator integrates Agent 1 and Agent 2  
- ✋ Human approval loop is fully functional  
- 🔄 Dynamic tool discovery verified  
- 🧠 Agent-to-agent architecture completed  
- 🚧 Full automation pending further validation  

---

## 🧾 Summary

This project demonstrates a scalable, protocol-driven multi-agent system with strong safety and extensibility guarantees.  
By separating reasoning, execution, and approval, the architecture supports complex AI workflows suitable for enterprise-grade applications.

🚀 Designed for advanced agent orchestration, RAG systems, and controlled AI execution.
