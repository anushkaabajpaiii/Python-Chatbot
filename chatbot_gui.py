import os
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract question-answer pairs from a docx file
def extract_faq_pairs(doc_path):
    doc = docx.Document(doc_path)
    faq_pairs = []
    question, answer = "", ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("Q:"):
            if question and answer:
                faq_pairs.append((question, answer))
                answer = ""
            question = text[2:].strip()
        elif text.startswith("A:"):
            answer = text[2:].strip()
        elif answer:
            answer += " " + text
    if question and answer:
        faq_pairs.append((question, answer))
    return faq_pairs

# Create FAISS index
def create_faiss_index(questions):
    embeddings = embedding_model.encode(questions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Search best answer
def search(query, questions, answers, index):
    if not query.strip():
        return "Please enter a question."
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), 1)
    best_idx = I[0][0]
    best_score = D[0][0]
    if best_score > 1.5:
        return "Sorry, I couldn't find a relevant answer. Try rephrasing."
    return answers[best_idx]

# Load and prepare the chatbot
doc_path = "sample_faq.docx"  # Make sure this file exists in the same directory
if not os.path.exists(doc_path):
    raise FileNotFoundError(f"FAQ file not found: {doc_path}")

faq_pairs = extract_faq_pairs(doc_path)
questions = [q for q, a in faq_pairs]
answers = [a for q, a in faq_pairs]
index, _ = create_faiss_index(questions)

# Gradio chatbot function
def chatbot_response(message):
    return search(message, questions, answers, index)

# Gradio interface
demo = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(placeholder="Ask me anything from the FAQ..."),
    outputs="text",
    title="📘 FAQ Chatbot",
    description="Ask a question and get instant answers from the embedded FAQ document!",
)

demo.launch()
