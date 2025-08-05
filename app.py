import streamlit as st

st.set_page_config(page_title="BrainEase", layout="wide")

from langchain_openai import ChatOpenAI
from utils.constants import api_key, LLM_MODEL

# Initialize LLM and store in session state
if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)


def add_header():
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <h1>🧠 BrainEase </h1>
        </div>
    """, unsafe_allow_html=True)


def home_page():
    add_header()
    st.markdown("""
        ### Welcome to BrainEase

        This app helps you navigate your migraines, giving you the support you need. You can:

        - Upload relevant medical documents or migraine diaries
        - Ask questions via text or voice 
        - Evaluate the assistant’s performance 

        ---
        Select a feature from the sidebar to get started!
    """)


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
    st.Page(home_page, title="Home Page"),
    st.Page(chatbot_page, title="Migraine Assistant"),
    st.Page(upload_pdfs_page, title="Upload Documents"),
    st.Page(evaluate_page, title="Evaluate"),
])

page.run()