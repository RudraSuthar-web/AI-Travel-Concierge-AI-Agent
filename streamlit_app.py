import streamlit as st
import os
from app.ui.document_manager import render_document_manager
from app.ui.chat import render_chat_interface
from app.rag.vectorstore import load_vector_store

# 1. Page Configuration
st.set_page_config(
    page_title="AI Travel Concierge AI Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapses the sidebar container by default
)

# Load and inject custom CSS style overrides
css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "ui", "style.css")
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Session State Initialization
if "documents" not in st.session_state:
    st.session_state["documents"] = []

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

if "vector_store" not in st.session_state:
    # Attempt to restore FAISS vector db from local cache if present
    st.session_state["vector_store"] = load_vector_store()

# 3. Render Dashboard Components (Single-Column)
render_document_manager()     # Collapsible Document Manager at the top
render_chat_interface()       # Standard centered Chat bubbles
