import os
import streamlit as st

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
# Helper: Process a single PDF file
# -----------------------------
def process_pdf_to_documents(pdf_path: str, llm) -> list[Document]:
    docs = []
    base64_images = pdf_to_base64_images(pdf_path)

    for i, img in enumerate(base64_images):
        with st.spinner(f"{pdf_path} - Extracting text from page {i+1}..."):
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
    try:
        os.remove(BUFFER_DOCS_PATH)
    except FileNotFoundError:
        pass

    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    all_docs = []

    for pdf_file in pdf_files:
        path = os.path.join(UPLOAD_DIR, pdf_file)
        st.info(f"Processing `{pdf_file}`")
        docs = process_pdf_to_documents(path, llm)
        all_docs.extend(docs)

    save_to_pickle(all_docs, BUFFER_DOCS_PATH)
    st.success(f"Indexed {len(all_docs)} PDF page(s).")

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

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        if st.button("Save PDF"):
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"File saved to: `{save_path}`")

    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    if pdf_files:
        st.write(f"Found {len(pdf_files)} PDF file(s):")
        for file in sorted(pdf_files):
            st.markdown(f"- 📄 `{file}`")
    else:
        st.info("No PDF files found in the upload folder.")

    if st.button("Index PDF Files"):
        call_index(llm)

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    from utils.constants import api_key, LLM_MODEL

    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
    main_index(llm)
