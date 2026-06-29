import streamlit as st
import os
from app.config import (
    GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, MAX_FILE_SIZE_MB
)
from app.rag.loaders import load_document_from_bytes
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vector_store, save_vector_store, clear_vector_store
from app.utils.helpers import allowed_file, validate_file_size
from app.logger import get_logger

log = get_logger("ui.document_manager")

def render_document_manager():
    """
    Renders the collapsible Document Manager and Database Control card at the top
    of the main page.
    """
    # Create expander card
    with st.expander("💼 Document Manager & Settings", expanded=False):
        col1, col2 = st.columns([3, 2], gap="large")
        
        with col1:
            st.markdown("<h5 style='margin-bottom: 0.5rem;'>📄 Ingest Travel Documents</h5>", unsafe_allow_html=True)
            st.caption("Upload flight tickets, hotel reservations, or packing lists (PDF, TXT, DOCX).")
            
            uploaded_files = st.file_uploader(
                "Choose documents",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            # Ingestion pipeline trigger
            if uploaded_files:
                new_files_detected = False
                for f in uploaded_files:
                    if f.name not in st.session_state["processed_files"]:
                        new_files_detected = True
                        break
                        
                if new_files_detected:
                    with st.spinner("Parsing and indexing new documents..."):
                        all_new_documents = []
                        for uploaded_file in uploaded_files:
                            file_name = uploaded_file.name
                            file_bytes = uploaded_file.read()
                            
                            # Validate file format and size
                            if not allowed_file(file_name):
                                st.error(f"Unsupported format: {file_name}")
                                continue
                            if not validate_file_size(len(file_bytes), MAX_FILE_SIZE_MB):
                                st.error(f"File too large (> {MAX_FILE_SIZE_MB}MB): {file_name}")
                                continue
                                
                            try:
                                docs = load_document_from_bytes(file_bytes, file_name)
                                all_new_documents.extend(docs)
                                st.session_state["processed_files"].add(file_name)
                                log.info(f"Loaded {len(docs)} document pages from '{file_name}'")
                            except Exception as e:
                                st.error(f"Error parsing '{file_name}': {str(e)}")
                                
                        if all_new_documents:
                            st.session_state["documents"].extend(all_new_documents)
                            
                            # Chunk documents
                            chunks = split_documents(st.session_state["documents"])
                            st.session_state["chunks"] = chunks
                            
                            try:
                                # Re-create and cache vector store
                                vector_store = create_vector_store(chunks)
                                st.session_state["vector_store"] = vector_store
                                save_vector_store(vector_store)
                                st.success(f"Indexed {len(st.session_state['processed_files'])} documents successfully!")
                            except Exception as e:
                                st.error(f"Failed to create vector store: {str(e)}")
                                log.error(f"Vector store compilation failure: {str(e)}", exc_info=True)
                                
        with col2:
            st.markdown("<h5 style='margin-bottom: 0.5rem;'>📊 Ingestion Status</h5>", unsafe_allow_html=True)
            
            if st.session_state["processed_files"]:
                st.markdown(f"**Ingested Files:** `{len(st.session_state['processed_files'])}`")
                for file_name in sorted(st.session_state["processed_files"]):
                    st.markdown(f"- 📄 `{file_name}`")
                st.markdown(f"**Total Search Chunks:** `{len(st.session_state.get('chunks', []))}`")
            else:
                st.warning("No travel files indexed yet. The concierge is running in General Mode.")
                
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            
            # Database clear trigger
            if st.button("🗑️ Clear Database & Chat Thread", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["documents"] = []
                st.session_state["chunks"] = []
                st.session_state["processed_files"] = set()
                st.session_state["vector_store"] = None
                
                # Delete local FAISS cache directory
                clear_vector_store()
                
                st.success("Database and chat history cleared!")
                st.rerun()
