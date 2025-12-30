import os
import requests
import numpy as np
import faiss
import gradio as gr

from groq import Groq
from sentence_transformers import SentenceTransformer

# =====================================
# Load Groq API Key (Hugging Face)
# =====================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

client = Groq(api_key=GROQ_API_KEY)

# =====================================
# Embedding Model (FREE)
# =====================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# Knowledge Base (Google Drive links)
# =====================================
DOCUMENT_LINKS = [
    "https://drive.google.com/uc?id=16VB3zfP9cBoP1jI_2FXk74fHDCN2Pzif"
]

def load_knowledge_base():
    texts = []
    for url in DOCUMENT_LINKS:
        response = requests.get(url)
        response.raise_for_status()
        texts.append(response.text)
    return "\n".join(texts)

# =====================================
# Build FAISS Vector Store (ONCE)
# =====================================
def build_vector_store():
    raw_text = load_knowledge_base()

    chunk_size = 500
    chunks = [
        raw_text[i:i + chunk_size]
        for i in range(0, len(raw_text), chunk_size)
    ]

    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks

faiss_index, stored_chunks = build_vector_store()

# =====================================
# RAG Question Answering
# =====================================
def ask_question(question):
    if not question.strip():
        return "**Please enter a question.**"

    query_embedding = embedder.encode([question]).astype("float32")
    _, indices = faiss_index.search(query_embedding, k=4)

    context = "\n\n".join(stored_chunks[i] for i in indices[0])

    prompt = f"""
You are a retrieval-based assistant.
Answer ONLY using the information provided in the context.
If the answer is not present, say exactly:
"I don’t know. This information is not available in my knowledge base."
Context:
{context}
Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# =====================================
# UI (HCI-Oriented Clean Design)
# =====================================
css = """
body {
    background-color: #020617;
    color: #ffffff;
    font-family: Inter, system-ui, sans-serif;
}
h1 {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 0 12px rgba(99,102,241,0.8);
}
.subtitle {
    text-align: center;
    color: #a5b4fc;
    margin-bottom: 20px;
}
textarea {
    background-color: #020617 !important;
    color: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid #6366f1 !important;
}
button {
    background: linear-gradient(to right, #6366f1, #22d3ee) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
}
.answer-box {
    background-color: #020617;
    border: 1px solid #6366f1;
    border-radius: 14px;
    padding: 16px;
    margin-top: 14px;
    color: blue;
    box-shadow: 0 0 15px rgba(99,102,241,0.4);
}
"""

# =====================================
# Gradio App
# =====================================
with gr.Blocks(css=css) as demo:
    gr.Markdown("# 📘 Knowledge Base Assistant")
    gr.Markdown(
        "<div class='subtitle'>Ask questions strictly from the connected documents</div>"
    )

    question = gr.Textbox(
        label="Your Question",
        placeholder="Type your question here...",
        lines=2
    )

    ask_btn = gr.Button("Get Answer")

    answer = gr.Markdown(
        value="**Answer will appear here**",
        elem_classes="answer-box"
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=answer
    )

demo.launch()
