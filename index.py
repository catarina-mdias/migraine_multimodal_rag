import os
import streamlit as st
from PIL import Image
import base64
import io

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
# Helper: Process a single image file (.jpg)
# -----------------------------
def process_image_to_document(image_path: str, llm) -> Document:
    """Convert a JPG image to a LangChain Document using OCR + LLM."""
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

    with st.spinner(f"{image_path} - Extracting text..."):
        text = base64_image_to_markdown(base64_str, llm)

    return Document(
        page_content=text,
        metadata={"source": os.path.basename(image_path)}
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

    files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".pdf", ".jpg", ".jpeg"))]
    all_docs = []

    for file in files:
        path = os.path.join(UPLOAD_DIR, file)
        st.info(f"Processing `{file}`")

        if file.lower().endswith(".pdf"):
            docs = process_pdf_to_documents(path, llm)
            all_docs.extend(docs)
        elif file.lower().endswith((".jpg", ".jpeg")):
            doc = process_image_to_document(path, llm)
            all_docs.append(doc)

    save_to_pickle(all_docs, BUFFER_DOCS_PATH)
    st.success(f"Indexed {len(all_docs)} document page(s).")

# -----------------------------
# Streamlit Interface
# -----------------------------
def main_index(llm):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.title("📄 PDF & JPG Upload and Indexing - Tutai")

    uploaded_file = st.file_uploader("Upload a PDF or JPG file", type=["pdf", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        if st.button("Save File"):
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"File saved to: `{save_path}`")

    if os.path.exists(UPLOAD_DIR):
        files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".pdf", ".jpg", ".jpeg"))]
        if files:
            st.write(f"Found {len(files)} file(s):")
            for file in sorted(files):
                st.markdown(f"- 🗂️ `{file}`")
        else:
            st.info("No PDF or JPG files found in the upload folder.")
    else:
        st.warning(f"The folder `{UPLOAD_DIR}` does not exist.")

    if st.button("Index Files"):
        call_index(llm)

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    from utils.constants import api_key, LLM_MODEL

    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
    main_index(llm)
