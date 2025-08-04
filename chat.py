import os
import time
import base64
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain import hub

from utils.constants import (
    api_key, LLM_MODEL, EMBEDDING_MODEL, BUFFER_DOCS_PATH,
    TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
)
from utils.utils import (
    save_to_pickle, load_from_pickle, pdf_to_base64_images,
    base64_image_to_markdown
)

# Initialize LLM and Embeddings
llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
client = OpenAI(api_key=api_key)
vector_store = InMemoryVectorStore(embeddings)
prompt = hub.pull("rlm/rag-prompt")

# State for LangGraph
from typing_extensions import TypedDict, List

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    return {"context": vector_store.similarity_search(state["question"], k=TOP_K)}

def generate(state: State):
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": context_text})
    response = llm.invoke(messages)
    return {"answer": response.content}

def response_generator():
    for word in st.session_state.temp_answer.split():
        yield word + " "
        time.sleep(0.05)

def tts_util(input_text):
    from datetime import datetime

    # Create 'audio' directory if it doesn't exist
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    speech_file_path = os.path.join(audio_dir, f"answer_{timestamp}.mp3")

    # Create audio using OpenAI API
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=input_text
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path


def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>"""
        components.html(audio_html, height=100)

def stt_util(audio):
    if not audio:
        return ""
    return client.audio.transcriptions.create(model="whisper-1", file=audio).text

def main_chat():
    st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
    st.title("Tutai Bot - Write and Speak!")

    # Show session state
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if 'temp_answer' not in st.session_state:
        st.session_state.temp_answer = ''
    if "show_voice" not in st.session_state:
        st.session_state.show_voice = False
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "indexing" not in st.session_state:
        st.session_state.indexing = False
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help you today?"}]

    st.checkbox("Voice enabled", key="show_voice")

    if st.button("Index and Vector Store document") and not st.session_state.indexing:
        with st.spinner("Loading and indexing..."):
            docs = load_from_pickle(BUFFER_DOCS_PATH)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents(docs)
            vector_store.add_documents(splits)
            graph = StateGraph(State).add_sequence([retrieve, generate])
            graph.add_edge(START, "retrieve")
            st.session_state.graph = graph.compile()
            st.session_state.indexing = True
            st.info(f"Indexed {len(splits)} document chunks.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("How can I help?"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            response = st.session_state.graph.invoke({"question": user_input})
            st.session_state.temp_answer = response['answer']
            st.write_stream(response_generator())
            with st.expander("Sources"):
                for d in response['context']:
                    st.write(f"**{d.metadata['source']}**")
                    st.markdown(d.page_content)
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

    if st.session_state.show_voice:
        audio = st.audio_input("Speak your question")

        # Only process new audio
        if audio and audio != st.session_state.last_audio:
            with st.spinner("Transcribing..."):
                prompt = stt_util(audio)

            st.markdown(f"**Transcript:** {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.last_audio = audio  # Mark audio as processed

            with st.spinner("Answering..."):
                response = st.session_state.graph.invoke({"question": prompt})
                st.session_state.temp_answer = response['answer']
                audio_path = tts_util(response['answer'])
                autoplay_audio(audio_path)

                with st.chat_message("assistant"):
                    st.write_stream(response_generator())
                    with st.expander("Sources"):
                        for d in response['context']:
                            st.write(f"**{d.metadata['source']}**")
                            st.markdown(d.page_content)
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})


if __name__ == '__main__':
    main_chat()
