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
def call_index(llm):
    existing_docs = load_existing_docs(BUFFER_DOCS_PATH)
    existing_sources = {doc.metadata.get("source").split(" - ")[0] for doc in existing_docs}

    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    all_new_docs = []

    for pdf_file in pdf_files:
        if pdf_file in existing_sources:
            continue  # Skip files already indexed

        path = os.path.join(UPLOAD_DIR, pdf_file)
        docs = process_pdf_to_documents(path, llm)
        all_new_docs.extend(docs)

    # Combine and save
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

    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Upload"):
            for uploaded_file in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            num_new_docs = call_index(llm)

            if num_new_docs > 0:
                st.success(f"Uploaded and indexed {len(uploaded_files)} new document(s).")
            else:
                st.info("No new PDFs to index. All documents are already processed.")

    # Display current uploaded PDFs
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    if pdf_files:
        st.markdown("### 🗂️ Current Documents:")
        for file in sorted(pdf_files):
            st.write(f"- {file}")
    else:
        st.info("No uploaded documents found.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    from utils.constants import api_key, LLM_MODEL

    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
    main_index(llm)
