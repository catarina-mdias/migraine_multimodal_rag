import os
import streamlit as st
import base64

from langchain_core.documents import Document

from utils.constants import (
    UPLOAD_DIR,
    BUFFER_DOCS_PATH
)

from utils.utils import (
    base64_image_to_markdown,
    save_to_pickle
)

# -----------------------------
# Helper: Process image to Document
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
# Indexing Entry Point
# -----------------------------
def call_index_images(llm):
    try:
        os.remove(BUFFER_DOCS_PATH)
    except FileNotFoundError:
        pass

    image_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    all_docs = []

    for image_file in image_files:
        path = os.path.join(UPLOAD_DIR, image_file)
        st.info(f"Processing `{image_file}`")
        doc = process_image_to_document(path, llm)
        all_docs.append(doc)

    save_to_pickle(all_docs, BUFFER_DOCS_PATH)
    st.success(f"Indexed {len(all_docs)} image(s).")

# -----------------------------
# Streamlit Interface
# -----------------------------
def main_image_index(llm):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.title("🖼️ Image Submission")

    st.markdown("""
This page allows you to **submit images related to migraines**, for example:
- Photos of medication instructions
- Ingredient lists
- Handwritten doctor notes

Text will be extracted from these images and indexed for retrieval.
""")

    uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        if st.button("Save Image"):
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"Image saved to: `{save_path}`")

    image_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    if image_files:
        st.write(f"Found {len(image_files)} image file(s):")
        for file in sorted(image_files):
            st.markdown(f"- 🖼️ `{file}`")
    else:
        st.info("No JPG files found in the upload folder.")

    if st.button("Index Image Files"):
        call_index_images(llm)
