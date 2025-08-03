import streamlit as st

st.set_page_config(page_title="Migraine Assistant", layout="wide")

from langchain_openai import ChatOpenAI
from utils.constants import api_key, LLM_MODEL

# Initialize LLM and store in session state
if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)


def add_header():
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <h1>🧠 Migraine Assistant</h1>
        </div>
    """, unsafe_allow_html=True)


# Wrapper functions with lazy imports
def upload_pdfs_page():
    from index import main_index
    add_header()
    main_index(st.session_state.llm)

# def upload_images_page():
#     from index_image import main_image_index
#     main_image_index(st.session_state.llm)

def chatbot_page():
    from chat import main_chat
    add_header()
    main_chat()

def evaluate_page():
    from rag_evaluate import main_eval
    add_header()
    main_eval()

# Navigation using Streamlit's new API
page = st.navigation([
    st.Page(upload_pdfs_page, title="📄 Upload PDFs"),
    # st.Page(upload_images_page, title="🖼️ Upload Images"),
    st.Page(chatbot_page, title="💬 Chatbot"),
    st.Page(evaluate_page, title="📊 Evaluate"),
])

page.run()