import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Create OpenAI client
client = OpenAI

# Load data and embeddings
df = pd.read_csv("chunks.csv")
chunks = df.apply(lambda row: row.to_json(), axis=1).tolist()
embeddings = np.load("embeddings.npy")

# Load embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Loan Dataset RAG Q&A Chatbot")

question = st.text_input("Ask your question about the loan data:")

if question:
    # Embed the question
    q_emb = embedder.encode([question])[0]

    # Compute cosine similarity
    scores = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))

    # Get top 3 relevant chunks
    top_k = 3
    top_idx = np.argsort(scores)[-top_k:][::-1]

    context = "\n\n".join([chunks[i] for i in top_idx])

    prompt = f"""Answer the question using ONLY the context below.
Context:
{context}

Question: {question}

Answer:
"""

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write("Answer:", response.choices[0].message.content)

