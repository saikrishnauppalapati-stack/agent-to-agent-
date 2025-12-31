import sys
import json
import logging
import os

# Configure logging to stderr so it doesn't interfere with JSON-RPC on stdout
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-kb-server")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    logger.error(f"Missing dependencies: {e}. Please install langchain-community, faiss-cpu, langchain-huggingface, sentence-transformers")
    sys.exit(1)

# --- Knowledge Base & RAG Setup ---

def initialize_knowledge_base():
    """
    Initializes the vector store with sample data.
    In a real app, you would load PDFs or text files here.
    """
    logger.info("Initializing Knowledge Base...")
    
    # 1. Create Sample Documents (Task: Create a small knowledge base)
    # You can replace this with DirectoryLoader to load PDFs/Text files
    raw_documents = [
        Document(page_content="The remote work policy allows employees to work from home up to 3 days a week. Approval from the manager is required.", metadata={"source": "policy_manual.txt", "topic": "Remote Work"}),
        Document(page_content="Expense reimbursements must be submitted by the 25th of each month. Receipts are required for any expense over $50.", metadata={"source": "finance_policy.txt", "topic": "Expenses"}),
        Document(page_content="The annual company retreat is scheduled for September 15th at the Grand Hotel. All employees are expected to attend.", metadata={"source": "events_memo.txt", "topic": "Events"}),
        Document(page_content="To reset your VPN password, visit the IT portal at portal.company.com and select 'Forgot Password'.", metadata={"source": "it_faq.txt", "topic": "IT Support"}),
        Document(page_content="Employees are eligible for health insurance benefits after 30 days of employment. The company covers 80% of the premium.", metadata={"source": "benefits_guide.txt", "topic": "Health Insurance"}),
        Document(page_content="All code changes must be reviewed by at least one peer before merging to the main branch. Use the pull request template provided.", metadata={"source": "engineering_handbook.txt", "topic": "Code Review"}),
        Document(page_content="The office will be closed for the following holidays: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, and Christmas Day.", metadata={"source": "holiday_schedule.txt", "topic": "Holidays"}),
        Document(page_content="Lost security badges must be reported immediately to the security desk. A replacement fee of $20 applies.", metadata={"source": "security_policy.txt", "topic": "Security"}),
    ]

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(raw_documents)

    # 3. Generate Embeddings & Store in Vector DB (Task: Embeddings & Store in Chroma)
    # Using a local model to avoid API costs for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ephemeral FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    logger.info(f"Knowledge Base initialized with {len(splits)} chunks.")
    return vectorstore, splits

# Global reference to the vector store and documents
vectorstore = None
all_documents = []

# --- MCP Server Protocol Handling ---

def handle_request(request):
    global vectorstore, all_documents
    method = request.get("method")
    params = request.get("params", {})
    
    # Initialize Request
    if method == "initialize":
        vectorstore, all_documents = initialize_knowledge_base()
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "KnowledgeBaseServer", "version": "1.0"}
        }
    
    # List Available Tools
    elif method == "tools/list":
        return {
            "tools": [
                {
                    "name": "search_knowledge_base",
                    "description": "Searches the internal knowledge base (policies, FAQs, docs) for relevant information.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query or question to look up."
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "list_documents",
                    "description": "Lists all documents available in the knowledge base.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "read_document",
                    "description": "Retrieves the full content of a specific document. Useful for summarization.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "The source filename of the document to read."}
                        },
                        "required": ["source"]
                    }
                }
            ]
        }
    
    # Call a Tool
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "search_knowledge_base":
            if not vectorstore:
                return {"content": [{"type": "text", "text": "Error: Knowledge base not initialized."}]}
            
            query = args.get("query", "")
            # Task: Implement retrieval function (Input: query, Output: top-k chunks)
            results = vectorstore.similarity_search(query, k=3)
            
            # Format the output for the agent
            context_text = "\n\n".join([f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in results])
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": context_text if context_text else "No relevant information found."
                    }
                ]
            }
        elif name == "list_documents":
            if not vectorstore:
                return {"content": [{"type": "text", "text": "Error: Knowledge base not initialized."}]}
            
            # Retrieve metadata from stored documents list
            unique_sources = {doc.metadata.get("source") for doc in all_documents if doc.metadata.get("source")}
            
            return {
                "content": [{
                    "type": "text", 
                    "text": "Available Documents:\n" + "\n".join(f"- {s}" for s in sorted(unique_sources))
                }]
            }
        elif name == "read_document":
            if not vectorstore:
                return {"content": [{"type": "text", "text": "Error: Knowledge base not initialized."}]}
            
            source = args.get("source")
            
            # Filter documents by source
            relevant_docs = [doc.page_content for doc in all_documents if doc.metadata.get("source") == source]
            
            if not relevant_docs:
                # If not found, list available sources to help the agent correct itself
                unique_sources = {doc.metadata.get("source") for doc in all_documents if doc.metadata.get("source")}
                return {"content": [{"type": "text", "text": f"No document found with source: '{source}'. Available sources: {', '.join(sorted(unique_sources))}"}]}
            
            return {"content": [{"type": "text", "text": "\n\n".join(relevant_docs)}]}
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    elif method == "notifications/initialized":
        return None
    
    return None

def main():
    # Read JSON-RPC messages from stdin
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = handle_request(request)
            
            # Only send response if it's a request (has 'id'), notifications don't get responses
            if "id" in request:
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": result
                }
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            if "id" in request:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32603, "message": str(e)}
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

if __name__ == "__main__":
    main()