import streamlit as st
from index import main_index
from chat import main_chat
from rag_evaluate import main_eval

from langchain_openai import ChatOpenAI
from utils.constants import api_key, LLM_MODEL

llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)

# Sidebar navigation
st.sidebar.title("Migraine Assistant")
page = st.sidebar.radio("Choose a page", ["📄 Upload & Index", "💬 Chatbot", "📊 Evaluate"])

# Load environment variables, setup, shared configs
# ... (add shared imports, setup OpenAI key, models, vector store, etc.)

# Load content based on selected page
if page == "📄 Upload & Index":
    main_index(llm)
elif page == "💬 Chatbot":
    main_chat()
elif page == "📊 Evaluate":
    main_eval()
