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
from mutagen.mp3 import MP3

from utils.constants import (
    api_key, LLM_MODEL, EMBEDDING_MODEL, BUFFER_DOCS_PATH,
    TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
)
from utils.utils import (
    save_to_pickle, load_from_pickle, pdf_to_base64_images,
    base64_image_to_markdown, autoplay_audio
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


# def main_chat():
#     st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
#     st.title("Migraine Assistant")
#
#     # Initialize session state variables
#     if 'temp_answer' not in st.session_state:
#         st.session_state.temp_answer = ''
#     if "graph" not in st.session_state:
#         st.session_state.graph = None
#     if "indexing" not in st.session_state:
#         st.session_state.indexing = False
#     if "messages" not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help you today?"}]
#     if "input_mode" not in st.session_state:
#         st.session_state.input_mode = "text"
#     if "last_processed_audio" not in st.session_state:
#         st.session_state.last_processed_audio = None
#     if "ready_for_next_audio" not in st.session_state:
#         st.session_state.ready_for_next_audio = True
#     if "audio_files" not in st.session_state:
#         st.session_state.audio_files = []
#     if "processing_audio" not in st.session_state:
#         st.session_state.processing_audio = False
#     if "show_audio_input" not in st.session_state:
#         st.session_state.show_audio_input = True
#     if "audio_mode_locked" not in st.session_state:
#         st.session_state.audio_mode_locked = False
#     if "recording_state" not in st.session_state:
#         st.session_state.recording_state = "ready"  # ready, recording, processing
#     if "current_audio_input" not in st.session_state:
#         st.session_state.current_audio_input = None
#
#     # After rerun from delayed rerun, show record button for subsequent messages
#     if not st.session_state.processing_audio and not st.session_state.show_audio_input:
#         user_message_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
#         if user_message_count > 0:  # Not the first message
#             st.session_state.recording_state = "show_button"
#         else:  # First message
#             st.session_state.show_audio_input = True
#             st.session_state.recording_state = "ready"
#
#     # Handle the case when audio finishes playing
#     if st.session_state.recording_state == "waiting_for_audio_end":
#         user_message_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
#         if user_message_count > 0:  # Not the first message
#             st.session_state.recording_state = "show_button"
#         else:  # First message
#             st.session_state.show_audio_input = True
#             st.session_state.recording_state = "ready"
#
#     # Controls row with reset button and audio toggle
#     col1, col2 = st.columns([2, 1])
#
#     with col1:
#         if st.button("🔄 Reset Session"):
#             # Clear all session state except essential ones
#             keys_to_keep = ["graph", "indexing"]  # Keep these to avoid reloading
#             keys_to_clear = [key for key in st.session_state.keys() if key not in keys_to_keep]
#             for key in keys_to_clear:
#                 del st.session_state[key]
#             # Reset essential variables
#             st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help you today?"}]
#             st.session_state.input_mode = "text"
#             st.session_state.last_processed_audio = None
#             st.session_state.ready_for_next_audio = True
#             st.session_state.processing_audio = False
#             st.session_state.audio_files = []
#             st.session_state.show_audio_input = True
#             st.session_state.audio_mode_locked = False
#             st.session_state.recording_state = "ready"
#             st.session_state.current_audio_input = None
#             st.session_state.recording_state = "ready"  # Make sure it's set to ready for delayed rerun
#             st.rerun()
#
#     with col2:
#         # Keep toggle always available - don't lock it
#         audio_mode = st.toggle("🎙️ Audio Mode", value=(st.session_state.input_mode == "audio"))
#         if audio_mode != (st.session_state.input_mode == "audio"):
#             st.session_state.input_mode = "audio" if audio_mode else "text"
#             st.session_state.ready_for_next_audio = True
#             st.session_state.show_audio_input = True
#             st.session_state.recording_state = "ready"
#
#     # Auto index document on load
#     if not st.session_state.indexing:
#         with st.spinner("Loading..."):
#             docs = load_from_pickle(BUFFER_DOCS_PATH)
#             from langchain_text_splitters import RecursiveCharacterTextSplitter
#             splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#             splits = splitter.split_documents(docs)
#             vector_store.add_documents(splits)
#             graph = StateGraph(State).add_sequence([retrieve, generate])
#             graph.add_edge(START, "retrieve")
#             st.session_state.graph = graph.compile()
#             st.session_state.indexing = True
#
#     # Display message history
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             # For assistant messages in audio mode, show replay button
#             if (message["role"] == "assistant" and
#                     st.session_state.input_mode == "audio" and
#                     i < len(st.session_state.audio_files) and
#                     st.session_state.audio_files[i] and
#                     os.path.exists(st.session_state.audio_files[i])):
#
#                 col1, col2 = st.columns([0.1, 0.9])
#                 with col1:
#                     create_audio_player(st.session_state.audio_files[i], i)
#                 with col2:
#                     st.markdown(message["content"])
#             else:
#                 st.markdown(message["content"])
#
#     # Handle text input mode
#     if st.session_state.input_mode == "text":
#         if user_input := st.chat_input("How can I help?"):
#             with st.chat_message("user"):
#                 st.markdown(user_input)
#             st.session_state.messages.append({"role": "user", "content": user_input})
#
#             with st.chat_message("assistant"):
#                 response = st.session_state.graph.invoke({"question": user_input})
#                 st.session_state.temp_answer = response['answer']
#                 st.write_stream(response_generator())
#                 with st.expander("Sources"):
#                     for d in response['context']:
#                         st.write(f"**{d.metadata['source']}**")
#                         st.markdown(d.page_content)
#                 st.session_state.messages.append({"role": "assistant", "content": response['answer']})
#
#     # Handle audio input mode
#     else:
#         if st.session_state.processing_audio:
#             if st.session_state.last_processed_audio:
#                 with st.spinner("Transcribing..."):
#                     prompt_text = stt_util(st.session_state.last_processed_audio)
#
#                 if prompt_text.strip():  # Only proceed if we got valid text
#                     # Lock audio mode after first successful recording
#                     st.session_state.audio_mode_locked = True
#
#                     with st.chat_message("user"):
#                         st.markdown(prompt_text)
#                     st.session_state.messages.append({"role": "user", "content": prompt_text})
#
#                     with st.spinner("Generating answer..."):
#                         response = st.session_state.graph.invoke({"question": prompt_text})
#                         st.session_state.temp_answer = response['answer']
#
#                         # Generate and play audio
#                         audio_path = tts_util(response['answer'])
#                         autoplay_audio(audio_path)
#
#                         # Store audio file for replay functionality
#                         st.session_state.audio_files.append(audio_path)
#
#                         with st.chat_message("assistant"):
#                             st.write_stream(response_generator())
#
#                             with st.expander("Sources"):
#                                 for d in response['context']:
#                                     st.write(f"**{d.metadata['source']}**")
#                                     st.markdown(d.page_content)
#
#                         st.session_state.messages.append({"role": "assistant", "content": response['answer']})
#
#                         # Get audio duration and set delayed rerun
#                         audio_duration = get_audio_duration(audio_path)
#                         delay = audio_duration + 2  # Add 2 seconds buffer
#
#                         # Hide audio input box during playback
#                         st.session_state.show_audio_input = False
#                         st.session_state.recording_state = "show_button"
#
#                         # Reset processing state and trigger delayed rerun
#                         st.session_state.processing_audio = False
#                         st.session_state.last_processed_audio = None
#                         delayed_rerun(delay)
#                 else:
#                     # If transcription failed, reset to ready state
#                     st.session_state.processing_audio = False
#                     st.session_state.last_processed_audio = None
#                     st.session_state.recording_state = "ready"
#                     st.session_state.show_audio_input = True
#         else:
#             # Check if this is the first message (show audio input) or subsequent messages (show button)
#             user_message_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
#
#             if user_message_count == 0:
#                 # First message - show original audio input
#                 if st.session_state.show_audio_input:
#                     audio = st.audio_input("Speak now")
#
#                     if audio and audio != st.session_state.last_processed_audio:
#                         st.session_state.processing_audio = True
#                         st.session_state.last_processed_audio = audio
#                         st.session_state.show_audio_input = False
#                         st.rerun()
#                 else:
#                     st.markdown("⏳ Please wait for the assistant to finish speaking before recording your question.")
#             else:
#                 # Subsequent messages - show record button after audio finishes
#                 if st.session_state.recording_state == "show_button":
#                     col1, col2, col3 = st.columns([1, 2, 1])
#                     with col2:
#                         record_clicked = st.button("🎤 Record New Message", key="record_new", use_container_width=True)
#
#                     if record_clicked:
#                         st.session_state.recording_state = "ready"
#                         st.session_state.show_audio_input = True
#                         st.rerun()
#
#                 elif st.session_state.show_audio_input:
#                     audio = st.audio_input("Speak now")
#
#                     if audio and audio != st.session_state.last_processed_audio:
#                         st.session_state.processing_audio = True
#                         st.session_state.last_processed_audio = audio
#                         st.session_state.show_audio_input = False
#                         st.rerun()
#                 else:
#                     st.markdown("⏳ Please wait for the assistant to finish speaking before recording your question.")

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
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ["graph", "indexing"]:
                    del st.session_state[key]
            st.rerun()

    with col2:
        audio_mode = st.toggle("🎙️ Audio Mode", value=(st.session_state.input_mode == "audio"))
        if not st.session_state.audio_mode_locked:  # Allow toggle only before locking
            st.session_state.input_mode = "audio" if audio_mode else "text"

    # --- Load docs (once) ---
    if not st.session_state.indexing:
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
            record_clicked = st.button("🎤 Record New Message", key="record_new", use_container_width=True)

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