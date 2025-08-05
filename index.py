import os
import streamlit as st
import pickle
import base64
import time

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
# PDF: Process a single PDF file
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
# JPG: Process image to Document
# -----------------------------
def process_image_to_document(image_path: str, llm) -> Document:
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

    with st.spinner(f"{os.path.basename(image_path)} - Extracting text..."):
        text = base64_image_to_markdown(base64_str, llm)

    return Document(
        page_content=text,
        metadata={"source": os.path.basename(image_path)}
    )


# -----------------------------
# Unified Indexing
# -----------------------------
def call_index(llm, just_uploaded_files: list[str]):
    existing_docs = load_existing_docs(BUFFER_DOCS_PATH)
    existing_sources = {doc.metadata.get("source").split(" - ")[0] for doc in existing_docs}

    all_new_docs = []

    for file_name in just_uploaded_files:
        if file_name in existing_sources:
            continue  # Skip already indexed

        path = os.path.join(UPLOAD_DIR, file_name)

        if file_name.lower().endswith(".pdf"):
            docs = process_pdf_to_documents(path, llm)
        elif file_name.lower().endswith((".jpg", ".jpeg")):
            doc = process_image_to_document(path, llm)
            docs = [doc]
        else:
            continue  # Unsupported file type

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
    st.title("Upload Documents")

    st.markdown("""
This page allows you to **submit medical PDFs and JPG images related to migraines**, such as:
- Clinical summaries
- Lab results
- Medication instruction leaflets
- Photos of handwritten notes or medication labels
""")

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, JPG)",
        type=["pdf", "jpg", "jpeg"],
        accept_multiple_files=True
    )

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
                st.info("All uploaded files were already indexed.")

    # Always show file list
    st.markdown("### Current Uploaded Files:")

    all_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".pdf", ".jpg", ".jpeg"))]
    if all_files:
        for file in sorted(all_files):
            st.markdown(f"- `{file}`")
    else:
        st.info("No uploaded documents found.")

    # Delete section
    delete_disabled = len(all_files) == 0
    if st.button("Delete All Files", disabled=delete_disabled):
        if os.path.exists(BUFFER_DOCS_PATH):
            os.remove(BUFFER_DOCS_PATH)
        for file in all_files:
            os.remove(os.path.join(UPLOAD_DIR, file))
        st.warning("All documents and in-memory data have been deleted.")
        # time.sleep(1.2)
        # st.rerun()


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    from utils.constants import api_key, LLM_MODEL

    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
    main_index(llm)
