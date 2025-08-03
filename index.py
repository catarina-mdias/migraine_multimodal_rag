import os
import streamlit as st
import pickle

from langchain_core.documents import Document

from utils.constants import (
    UPLOAD_DIR,
    BUFFER_DOCS_PATH
)

from utils.utils import (
    pdf_to_base64_images,
    base64_image_to_markdown,
    save_to_pickle
)

# -----------------------------
# Load previously saved docs
# -----------------------------
def load_existing_docs(pickle_path):
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    return []

# -----------------------------
# Helper: Process a single PDF file
# -----------------------------
def process_pdf_to_documents(pdf_path: str, llm) -> list[Document]:
    docs = []
    base64_images = pdf_to_base64_images(pdf_path)

    for i, img in enumerate(base64_images):
        with st.spinner(f"{os.path.basename(pdf_path)} - Extracting text from page {i+1}..."):
            text = base64_image_to_markdown(img, llm)
            docs.append(Document(
                page_content=text,
                metadata={"source": f"{os.path.basename(pdf_path)} - Page {i+1}"}
            ))
    return docs

# -----------------------------
# Indexing Entry Point
# -----------------------------
def call_index(llm, just_uploaded_files: list[str]):
    existing_docs = load_existing_docs(BUFFER_DOCS_PATH)
    existing_sources = {doc.metadata.get("source").split(" - ")[0] for doc in existing_docs}

    all_new_docs = []

    for pdf_file in just_uploaded_files:
        if pdf_file in existing_sources:
            continue  # Skip already indexed
        path = os.path.join(UPLOAD_DIR, pdf_file)
        docs = process_pdf_to_documents(path, llm)
        all_new_docs.extend(docs)

    if all_new_docs:
        all_docs = existing_docs + all_new_docs
        save_to_pickle(all_docs, BUFFER_DOCS_PATH)

    return len(all_new_docs)

# -----------------------------
# Streamlit Interface
# -----------------------------
def main_index(llm):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.title("📄 PDF Submission")
    st.markdown("""
This page allows you to **submit medical PDFs related to migraines**, such as:
- Clinical summaries
- Lab results
- Medication instruction leaflets

The assistant will extract relevant content from your PDFs for later search and analysis.
""")

    uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Upload"):
            uploaded_names = []

            for uploaded_file in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_names.append(uploaded_file.name)

            num_new_docs = call_index(llm, uploaded_names)

            if num_new_docs > 0:
                st.success(f"Uploaded {len(uploaded_names)} new document(s).")
            else:
                st.info("All uploaded documents were already indexed.")

        # --- Always show Current Documents header ---
    st.markdown("### 🗂️ Current Documents:")

    # --- Load current uploaded PDFs ---
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]

    if pdf_files:
        for file in sorted(pdf_files):
            st.markdown(f"- `{file}`")
    else:
        st.info("No uploaded documents found.")

    # --- Delete Documents Button (disabled if no PDFs) ---
    delete_disabled = len(pdf_files) == 0
    if st.button("🗑️ Delete Documents", disabled=delete_disabled):
        # Delete in-memory docs file
        if os.path.exists(BUFFER_DOCS_PATH):
            os.remove(BUFFER_DOCS_PATH)
        # Delete uploaded PDFs
        for file in pdf_files:
            os.remove(os.path.join(UPLOAD_DIR, file))

        st.warning("All documents and in-memory data have been deleted.")


# def main_index(llm):
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     st.title("📄 PDF Submission")
#     st.markdown("""
# This page allows you to **submit medical PDFs related to migraines**, such as:
# - Clinical summaries
# - Lab results
# - Medication instruction leaflets
#
# The assistant will extract relevant content from your PDFs for later search and analysis.
# """)
#
#     uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)
#
#     if uploaded_files:
#         if st.button("Upload"):
#             uploaded_names = []
#
#             for uploaded_file in uploaded_files:
#                 save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
#                 with open(save_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
#                 uploaded_names.append(uploaded_file.name)
#
#             num_new_docs = call_index(llm, uploaded_names)
#
#             if num_new_docs > 0:
#                 st.success(f"Uploaded and indexed {len(uploaded_names)} new document(s).")
#             else:
#                 st.info("All uploaded documents were already indexed.")
#
#     # Display current uploaded PDFs
#     st.markdown("### 🗂️ Current Documents:")
#
#     pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
#     if pdf_files:
#         for file in sorted(pdf_files):
#             st.markdown(f"- `{file}`")
#     else:
#         st.info("No uploaded documents found.")
#
#     # --- Delete Documents Section ---
#     if st.button("🗑️ Delete Documents"):
#         st.session_state.show_delete_confirm = True
#
#     if st.session_state.get("show_delete_confirm", False):
#         st.warning("Are you sure you want to delete all uploaded documents and the in-memory data? This action cannot be undone.")
#         col1, col2 = st.columns(2)
#
#         with col1:
#             if st.button("Confirm Delete"):
#                 # Delete in-memory docs file
#                 if os.path.exists(BUFFER_DOCS_PATH):
#                     os.remove(BUFFER_DOCS_PATH)
#                 # Delete uploaded PDFs
#                 for file in os.listdir(UPLOAD_DIR):
#                     if file.lower().endswith(".pdf"):
#                         os.remove(os.path.join(UPLOAD_DIR, file))
#                 st.success("All documents and in-memory data have been deleted.")
#                 st.session_state.show_delete_confirm = False
#
#         with col2:
#             if st.button("Cancel"):
#                 st.session_state.show_delete_confirm = False


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    from utils.constants import api_key, LLM_MODEL

    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
    main_index(llm)
