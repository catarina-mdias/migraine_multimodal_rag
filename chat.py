import os
import time
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain import hub
from mutagen.mp3 import MP3

from utils.constants import (
    api_key, LLM_MODEL, EMBEDDING_MODEL, BUFFER_DOCS_PATH,
    TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
)
from utils.utils import (
    load_from_pickle,
    autoplay_audio
)

# Initialize LLM and Embeddings
llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
client = OpenAI(api_key=api_key)
vector_store = InMemoryVectorStore(embeddings)
prompt = hub.pull("rlm/rag-prompt")

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
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    speech_file_path = os.path.join(audio_dir, f"answer_{timestamp}.mp3")

    with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=input_text
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path


def get_audio_duration(file_path):
    """Get audio file duration in seconds"""
    try:
        audio = MP3(file_path)
        return audio.info.length
    except:
        # Fallback: estimate based on file size (rough approximation)
        file_size = os.path.getsize(file_path)
        # Rough estimate: 1 MB ≈ 60 seconds for MP3
        return max(3, file_size / (1024 * 1024) * 60)


def delayed_rerun(delay_seconds):
    """Trigger a rerun after specified delay using JavaScript"""
    delay_ms = int(delay_seconds * 1000)
    components.html(f"""
        <script>
        setTimeout(function() {{
            window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
        }}, {delay_ms});
        </script>
    """, height=0)


def stt_util(audio):
    if not audio:
        return ""
    return client.audio.transcriptions.create(model="whisper-1", file=audio).text


def main_chat():
    st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
    st.title("Migraine Assistant")

    # --- Initialize session state ---
    default_messages = [{"role": "assistant", "content": "Hi there! How can I help you today?"}]

    defaults = {
        "messages": default_messages,
        "input_mode": "text",
        "audio_files": [],
        "audio_mode_locked": False,
        "recording_state": "ready",  # ready, recording, processing
        "current_audio_input": None,
        "last_processed_audio": None,
        "processing_audio": False,
        "show_audio_input": False,
        "indexing": False,
        "graph": None,
        "temp_answer": '',
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- Reset / Audio Mode Toggle ---
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ["graph", "indexing"]:
                    del st.session_state[key]
            st.rerun()

    with col2:
        audio_mode = st.toggle("Audio Mode", value=(st.session_state.input_mode == "audio"))
        if not st.session_state.audio_mode_locked:  # Allow toggle only before locking
            st.session_state.input_mode = "audio" if audio_mode else "text"

    # --- Load docs (once) ---
    if not st.session_state.indexing:
        if not os.path.exists(BUFFER_DOCS_PATH):
            st.warning("⚠️ No documents found. Please go to the **Upload PDFs** page and upload your documents first.")
            st.stop()
        else:
            with st.spinner("Loading..."):
                docs = load_from_pickle(BUFFER_DOCS_PATH)
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                splits = splitter.split_documents(docs)
                vector_store.add_documents(splits)
                graph = StateGraph(State).add_sequence([retrieve, generate])
                graph.add_edge(START, "retrieve")
                st.session_state.graph = graph.compile()
                st.session_state.indexing = True

    # --- Message history ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- TEXT MODE ---
    if st.session_state.input_mode == "text":
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

    # --- AUDIO MODE ---
    else:
        user_message_count = len([m for m in st.session_state.messages if m["role"] == "user"])

        # Always show record button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            record_clicked = st.button("Record New Message", key="record_new", use_container_width=True)

        if record_clicked:
            st.session_state.recording_state = "recording"
            st.session_state.show_audio_input = True
            st.rerun()

        if st.session_state.recording_state == "recording" and st.session_state.show_audio_input:
            audio = st.audio_input("Speak now")

            if audio and audio != st.session_state.last_processed_audio:
                st.session_state.processing_audio = True
                st.session_state.last_processed_audio = audio
                st.session_state.recording_state = "processing"
                st.session_state.show_audio_input = False
                st.rerun()

        if st.session_state.processing_audio:
            with st.spinner("Transcribing..."):
                prompt_text = stt_util(st.session_state.last_processed_audio)

            if prompt_text.strip():
                st.session_state.audio_mode_locked = True

                with st.chat_message("user"):
                    st.markdown(prompt_text)
                st.session_state.messages.append({"role": "user", "content": prompt_text})

                with st.spinner("Generating answer..."):
                    response = st.session_state.graph.invoke({"question": prompt_text})
                    st.session_state.temp_answer = response['answer']
                    audio_path = tts_util(response['answer'])
                    autoplay_audio(audio_path)
                    st.session_state.audio_files.append(audio_path)

                    with st.chat_message("assistant"):
                        st.write_stream(response_generator())
                        with st.expander("Sources"):
                            for d in response['context']:
                                st.write(f"**{d.metadata['source']}**")
                                st.markdown(d.page_content)

                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                audio_duration = get_audio_duration(audio_path)
                delay = audio_duration + 2

                st.session_state.recording_state = "ready"
                st.session_state.processing_audio = False
                st.session_state.last_processed_audio = None

                delayed_rerun(delay)
            else:
                st.session_state.recording_state = "ready"
                st.session_state.processing_audio = False
                st.session_state.last_processed_audio = None
                st.session_state.show_audio_input = False


if __name__ == '__main__':
    main_chat()