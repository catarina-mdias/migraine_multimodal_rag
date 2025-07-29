import os
from dotenv import load_dotenv

# -------------------------
# Paths & Directories
# -------------------------
UPLOAD_DIR = "data"
BUFFER_DIR = "buffer"
BUFFER_DOCS_PATH = os.path.join(BUFFER_DIR, "docs.pkl")

# -------------------------
# OpenAI & Model Settings
# -------------------------
load_dotenv(dotenv_path=".env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o-mini"
LLM_EVAL_MODEL = "gpt-4.1-mini"  # used in eval script
EMBEDDING_MODEL = "text-embedding-3-large"

# -------------------------
# Vector Store Settings
# -------------------------
TOP_K = 5
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 2000

# -------------------------
# Audio Settings
# -------------------------
TTS_OUTPUT_PATH = "answer.mp3"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

# -------------------------
# Prompt for OCR to Markdown
# -------------------------
QUERY_OCR = """Extract all the text in the image as a markdown, including tables, headers and plain text.
    If you see any author or writer names, include a header saying "Authors" with the actual author information in it. If authors are not in text, don't include any header saying "Authors".
    If you find and image such as a diagram or other sort, create a description of the image.
    Do not use the word 'Markdown' or wrap the output in triple backticks. Avoid any code or markup formatting.
    markdown:
    """


