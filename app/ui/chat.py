import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.workflow import graph_app
from app.logger import get_logger

log = get_logger("ui.chat")

def render_chat_interface():
    """
    Renders the chat thread, reads user input, constructs graph input state,
    invokes the LangGraph agent, and outputs grounded answers with sources.
    """
    st.markdown("<h1 style='margin-bottom: 0;'>✈️ AI Travel Concierge Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic; color: gray; margin-top: 0;'>Secure, grounded RAG chatbot & itinerary planner (Weeks 1–2 Demo)</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Initialize conversational state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your **AI Travel Concierge Assistant** 🗺️✈️\n\n"
                    "I can answer general travel planning questions, or provide deep insights into your travel plans.\n\n"
                    "🚀 **How to use me:**\n"
                    "1. Expand the **Document Manager & Settings** at the top of the page and upload your flight confirmations, hotel reservations, or itineraries.\n"
                    "2. Ask questions about your trip details (e.g. *'What time is my flight?'* or *'What is my hotel reservation number?'*).\n"
                    "3. Try testing the custom agent nodes by asking me to **'summarize documents'** or **'list documents'**!"
                ),
                "sources": []
            }
        ]

    # 2. Render conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display source citations if present and role is assistant
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔍 Grounded References & Citations", expanded=False):
                    for i, doc in enumerate(msg["sources"]):
                        source_name = doc.metadata.get("source", "Uploaded Document")
                        page_num = doc.metadata.get("page", None)
                        page_suffix = f" (Page {page_num})" if page_num else ""
                        st.markdown(f"**Source {i+1}: {source_name}{page_suffix}**")
                        st.info(doc.page_content)

    # 3. Read and respond to user prompts
    if user_query := st.chat_input("Ask about your trip or uploaded documents..."):
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_query)
            
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # Display assistant thought / processing spinner
        with st.chat_message("assistant"):
            with st.spinner("Consulting Concierge Graph..."):
                # Convert session history into LangChain messages for state context
                history_messages = []
                for m in st.session_state["messages"]:
                    if m["role"] == "user":
                        history_messages.append(HumanMessage(content=m["content"]))
                    else:
                        history_messages.append(AIMessage(content=m["content"]))

                # Prepare the initial LangGraph state input dictionary
                graph_input = {
                    "messages": history_messages,
                    "query": user_query,
                    "documents": st.session_state.get("documents", []),
                    "retrieved_context": "",
                    "source_documents": [],
                    "routing_decision": "direct",
                    "tool_output": None,
                    "response": "",
                    "metrics": {}
                }

                try:
                    # Invoke LangGraph
                    output_state = graph_app.invoke(graph_input)
                    
                    response_content = output_state.get("response", "I apologize, but I could not formulate a response.")
                    source_docs = output_state.get("source_documents", [])
                    metrics_logged = output_state.get("metrics", {})
                    routing = output_state.get("routing_decision", "direct")
                    
                    st.markdown(response_content)
                    
                    # Display citation expander if sources are returned
                    if source_docs:
                        with st.expander("🔍 Grounded References & Citations", expanded=False):
                            for i, doc in enumerate(source_docs):
                                source_name = doc.metadata.get("source", "Uploaded Document")
                                page_num = doc.metadata.get("page", None)
                                page_suffix = f" (Page {page_num})" if page_num else ""
                                st.markdown(f"**Source {i+1}: {source_name}{page_suffix}**")
                                st.info(doc.page_content)
                                
                    # Display performance metrics hooks in sidebar/logs
                    log.info(f"Graph query successful. Routing: {routing}. Metrics: {metrics_logged}")
                    
                    # Append answer to session state history
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response_content,
                        "sources": source_docs
                    })
                    
                except Exception as e:
                    error_answer = f"⚠️ I encountered an error executing the workflow: {str(e)}"
                    st.error(error_answer)
                    log.error(f"LangGraph execution exception: {str(e)}", exc_info=True)
                    
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_answer,
                        "sources": []
                    })
