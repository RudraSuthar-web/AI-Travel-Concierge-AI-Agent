import time
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.config import LLM_PROVIDER, LLM_MODEL, GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY
from app.graph.state import AgentState
from app.rag.retriever import retrieve_relevant_documents
from app.tools.document_tools import summarize_corpus_tool, list_documents_tool
from app.logger import get_logger

log = get_logger("graph.nodes")

def get_llm():
    """
    Dynamically configures and returns the requested LLM client.
    """
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Set temperature to 0 for deterministic RAG answers
        return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
        
    elif LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment.")
        from langchain_groq import ChatGroq
        return ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0)
        
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user request to determine the execution path: 'tool', 'rag', or 'direct'.
    """
    query = state.get("query", "").lower().strip()
    docs = state.get("documents", [])
    
    log.info(f"Classifying user query: '{query}'")
    
    # Heuristics for document tools
    tool_keywords = ["summarize", "summary", "outline the documents", "corpus overview"]
    list_keywords = ["list files", "show files", "what documents", "show uploaded", "list documents"]
    
    if any(kw in query for kw in tool_keywords):
        log.info("Query matched summary tool routing.")
        return {"routing_decision": "tool", "metrics": {"tool_to_run": "summarize"}}
        
    if any(kw in query for kw in list_keywords):
        log.info("Query matched listing tool routing.")
        return {"routing_decision": "tool", "metrics": {"tool_to_run": "list"}}
        
    # If no documents are uploaded, we cannot run RAG. Route directly to chat model.
    if not docs:
        log.info("No documents uploaded. Routing to direct chat.")
        return {"routing_decision": "direct"}
        
    log.info("Documents found. Routing to RAG engine.")
    return {"routing_decision": "rag"}


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves document chunks relevant to the user query and injects them into state.
    """
    query = state.get("query", "")
    log.info(f"Executing retriever node for query: '{query}'")
    
    start_time = time.perf_counter()
    retrieved_chunks = retrieve_relevant_documents(query)
    duration = time.perf_counter() - start_time
    
    # Formulate context block
    context_text = ""
    for i, doc in enumerate(retrieved_chunks):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        page_str = f" (Page {page})" if page else ""
        context_text += f"--- Chunk {i+1} from {source}{page_str} ---\n{doc.page_content}\n\n"
        
    log.info(f"Retrieved {len(retrieved_chunks)} chunks in {duration:.4f}s")
    
    metrics = state.get("metrics", {})
    metrics["retrieval_time"] = duration
    
    return {
        "retrieved_context": context_text,
        "source_documents": retrieved_chunks,
        "metrics": metrics
    }


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs the selected document analysis tools.
    """
    docs = state.get("documents", [])
    metrics = state.get("metrics", {})
    tool_to_run = metrics.get("tool_to_run", "summarize")
    
    log.info(f"Executing tool node. Running tool: '{tool_to_run}'")
    
    if tool_to_run == "summarize":
        res = summarize_corpus_tool(docs)
        tool_output = res["summary_text"]
    else:
        tool_output = list_documents_tool(docs)
        
    return {
        "tool_output": tool_output,
        "response": tool_output
    }


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes responses based on LLM execution mode (direct chat vs RAG grounding).
    """
    query = state.get("query", "")
    decision = state.get("routing_decision", "direct")
    context = state.get("retrieved_context", "")
    messages = state.get("messages", [])
    
    llm = get_llm()
    
    start_time = time.perf_counter()
    
    if decision == "rag":
        log.info("Generating grounded response using RAG context...")
        system_prompt = (
            "You are an AI Travel Concierge Assistant. You are provided with travel details, "
            "itineraries, or documents uploaded by the user.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question relying strictly on the retrieved context below.\n"
            "2. If the answer is not present in the context, say: "
            "'I'm sorry, but I couldn't find that information in your uploaded documents.' "
            "and do not make up details or hypothesize.\n"
            "3. Maintain a helpful, polite travel concierge tone.\n\n"
            f"--- START RETRIEVED CONTEXT ---\n{context}\n--- END RETRIEVED CONTEXT ---"
        )
        
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
    else:
        log.info("Generating direct chat response...")
        system_prompt = (
            "You are an AI Travel Concierge Assistant. "
            "Help the user plan their travels, organize schedules, or answer questions.\n"
            "Let them know that they can expand the 'Document Manager & Settings' at the top of the page "
            "to upload travel documents (PDF, TXT, DOCX) and query specific itineraries directly!"
        )
        
        # Build conversational prompt
        prompt = [SystemMessage(content=system_prompt)]
        # Add past 4 messages for memory (excluding system messages)
        chat_history = []
        for msg in messages[-4:]:
            chat_history.append(msg)
        
        # Ensure latest query is added if not already at the end
        if not chat_history or chat_history[-1].content != query:
            chat_history.append(HumanMessage(content=query))
            
        prompt.extend(chat_history)
        
    try:
        res = llm.invoke(prompt)
        response_text = res.content
    except Exception as e:
        log.error(f"Error calling LLM: {str(e)}", exc_info=True)
        response_text = f"⚠️ I encountered an error communicating with the AI service: {str(e)}"
        
    duration = time.perf_counter() - start_time
    log.info(f"Answer generated in {duration:.4f}s")
    
    metrics = state.get("metrics", {})
    metrics["generation_time"] = duration
    
    return {
        "response": response_text,
        "metrics": metrics
    }
