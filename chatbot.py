import os
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract (question, answer) pairs from the docx file
def extract_faq_pairs(doc_path):
    doc = docx.Document(doc_path)
    faq_pairs = []
    question = ""
    answer = ""
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

# Search for best match
def search(query, questions, answers, index):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), 1)
    best_idx = I[0][0]
    best_score = D[0][0]

    # Optional: basic relevance filter (e.g., ignore very bad matches)
    if best_score > 1.5:
        return "Sorry, I couldn't find a relevant answer. Can you rephrase?"

    return answers[best_idx]

# Main loop
if __name__ == "__main__":
    doc_path = "sample_faq.docx"
    if not os.path.exists(doc_path):
        print(f"❌ File not found: {doc_path}")
        exit()

    # Step 1: Extract Q&A pairs
    faq_pairs = extract_faq_pairs(doc_path)
    questions = [q for q, a in faq_pairs]
    answers = [a for q, a in faq_pairs]

    # Step 2: Build FAISS index
    index, _ = create_faiss_index(questions)

    print("🤖 Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = search(user_input, questions, answers, index)
        print("Bot:", response)
