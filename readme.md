# 🧠 BrainEase - Multimodal RAG Migraine Assistant
BrainEase is a prototype of an intelligent assistant designed to support users in understanding and managing migraine-related content. Built with **multimodal RAG (Retrieval-Augmented Generation)** and integrated with **speech-to-text**, **text-to-speech**, and **image understanding**, BrainEase offers a natural, interactive chat experience where users can upload PDFs, talk to the assistant, and even show it visual information.


## Running the app locally
- Create and activate a virsutal environment: 
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
- Install dependencies
```
pip install -r requirements.txt
```

- Set your environment variables:you’ll need your OpenAI API key. Create a .env file in the root directory with this format (or copy .env_template):
```
OPENAI_API_KEY=your-openai-key
```

## Getting started with the app
**Step 1: Upload documents**
- Run the app: `streamlit run app.py`
- Navigate to the “Upload Documents” page in the sidebar.
- Upload one or more PDF documents or images. The system will process and index them.

**Step 2: Chat with the assistant**
- Switch to the “Migraine Assistant” page.
- Choose between text or audio mode.
- Ask your questions! The assistant will respond using information from your uploaded content.

**Step 3: Evaluate the Multimodal RAG System**
- Go to the “Evaluate” page.
- Upload a JSON file containing a list of test questions and expected answers.
- The app will:
  - Automatically generate answers using the current indexed documents. 
  - Display the results in a comparison table. 
  - Allow you to choose between:
    - Statistics-based evaluation (ROUGE-1, ROUGE-2, METEOR)
    - LLM-based evaluation (using GPT-based scoring of factual accuracy with reasoning)
  - You'll get per-question evaluation scores and an overall summary to assess system performance.


## Example Input Data
Sample documents and images for testing are available in the `input_data/` folder. You can use them directly without needing to upload your own files.
