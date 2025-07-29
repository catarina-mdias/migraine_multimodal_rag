import os
from dotenv import load_dotenv
import uuid
import base64
import io
import re
import pickle
import time
import json



import streamlit as st
import streamlit.components.v1 as components

## Providers
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Tuple
from langgraph.graph import START, StateGraph


def stt_util(audio):
    transcript = ''

    if audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )

    return transcript.text


def llm_completion(input_text):
    output_text = ''

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_text}],
        temperature=0,
    )

    output_text = response.choices[0].message.content

    return output_text


def tts_util(input_text):
    speech_file_path = "answer.mp3"

    # Check if the file exists, then remove it
    if os.path.exists(speech_file_path):
        os.remove(speech_file_path)

    with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=input_text
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path


def autoplay_audio(file_path: str):
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


def pdf_to_base64_images(pdf_path: str) -> list[str]:
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


def display_base64_image(base64_str, caption=""):
    image_html = f'<img src="data:image/png;base64,{base64_str}" width="800"/>'
    st.markdown(image_html, unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def clean_markdown_fences(text: str) -> str:
    # Remove triple backticks with optional language specifier
    cleaned = re.sub(r"```(?:\w+)?\n?", "", text)
    return cleaned.strip()


def base64_image_to_markdown(base64_str):
    query = """Extract all the text in the image as a markdown, including tables, headers and plain text.
    If you see any author or writer names, include a header saying "Authors"
    If you find and image such as a diagram or other sort, create a description of the image.
    Do not use the word 'Markdown' or wrap the output in triple backticks. Avoid any code or markup formatting.
    markdown:
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
            },
        ],
    )
    response_temp = llm.invoke([message])
    response = clean_markdown_fences(response_temp.content)

    return response


def save_to_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# Streamed response emulator
def response_generator():
    for word in st.session_state.temp_answer.split():
        yield word + " "
        time.sleep(0.05)
