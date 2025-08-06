import os
import pickle
import base64
import io
import re
import time
import fitz  # PyMuPDF
from PIL import Image
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from openai import OpenAI
import streamlit.components.v1 as components

from utils.constants import (
    TTS_OUTPUT_PATH,
    TTS_MODEL,
    TTS_VOICE,
    QUERY_OCR
)

# Initialize OpenAI client once
client = OpenAI()


# -------------------------
# Pickle Utilities
# -------------------------

def save_to_pickle(obj, filepath):
    """Save any object to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_from_pickle(filepath):
    """Load object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# -------------------------
# PDF & OCR Utilities
# -------------------------

def pdf_to_base64_images(pdf_path: str) -> list[str]:
    """Convert each page of a PDF to a base64-encoded image string."""
    pdf_document = fitz.open(pdf_path)
    base64_images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_image)

    return base64_images

def clean_markdown_fences(text: str) -> str:
    """Remove triple backticks and optional language specifier from markdown."""
    return re.sub(r"```(?:\w+)?\n?", "", text).strip()


def base64_image_to_markdown(base64_str: str, llm) -> str:
    image_message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_str}"  # or image/png if PNG
                }
            }
        ]
    )
    response = llm.invoke([image_message])
    return response.content


# -------------------------
# Audio Utilities
# -------------------------

def stt_util(audio) -> str:
    """Transcribe audio input to text using Whisper."""
    if not audio:
        return ""
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio
    )
    return transcript.text

def llm_completion(input_text: str) -> str:
    """Send a prompt to GPT and return its response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_text}],
        temperature=0,
    )
    return response.choices[0].message.content

def tts_util(input_text: str) -> str:
    """Convert text to speech and save to MP3."""
    if os.path.exists(TTS_OUTPUT_PATH):
        os.remove(TTS_OUTPUT_PATH)

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=input_text
    ) as response:
        response.stream_to_file(TTS_OUTPUT_PATH)

    return TTS_OUTPUT_PATH

def autoplay_audio(file_path: str):
    """Stream audio file with autoplay in Streamlit."""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

    audio_html = f"""
        <audio id="player" controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        <script>
            var audio = document.getElementById("player");
            audio.play();
        </script>
    """
    components.html(audio_html, height=100)


# -------------------------
# Streamlit Helpers
# -------------------------

def response_generator():
    """Yields word-by-word response streaming for UI."""
    for word in st.session_state.temp_answer.split():
        yield word + " "
        time.sleep(0.05)
