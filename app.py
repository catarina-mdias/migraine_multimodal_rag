import streamlit as st
from index import main_index
from index_image import main_image_index  # <-- new import
from chat import main_chat
from rag_evaluate import main_eval

from langchain_openai import ChatOpenAI
from utils.constants import api_key, LLM_MODEL

llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)

# Sidebar navigation
st.sidebar.title("Migraine Assistant")
page = st.sidebar.radio("Choose a page", [
    "📄 Upload PDFs",
    "🖼️ Upload Images",
    "💬 Chatbot",
    "📊 Evaluate"
])

# Load page content
if page == "📄 Upload PDFs":
    main_index(llm)
elif page == "🖼️ Upload Images":
    main_image_index(llm)  # <-- call the new one
elif page == "💬 Chatbot":
    main_chat()
elif page == "📊 Evaluate":
    main_eval()
