import streamlit as st
import os
import google.generativeai as genai

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


# -----------------------------
# Gemini Setup
# -----------------------------
API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

if not API_KEY:
    st.warning("Enter Gemini API Key")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# -----------------------------
# Custom Gemini Embedding Class
# -----------------------------
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            genai.embed_content(
                model="models/embedding-001",
                content=text
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(
            model="models/embedding-001",
            content=text
        )["embedding"]


embedding = GeminiEmbeddings()


# -----------------------------
# Load Crop Docs
# -----------------------------
def load_docs():
    docs = []

    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"docs/{file}")
            docs.extend(loader.load())

    return docs


documents = load_docs()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

db = FAISS.from_documents(chunks, embedding)


# -----------------------------
# Query Function
# -----------------------------
def answer_query(query):

    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are Kisan Saathi, an agriculture assistant for Indian farmers.

    Use the crop advisory information below to answer simply.

    Context:
    {context}

    Farmer Question:
    {query}

    Give short practical answer in simple language.
    Hindi allowed if query is Hindi.
    """

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# UI
# -----------------------------
st.title("🌾 Agriculture Kisan Saathi")
st.subheader("AI Crop Advisory Assistant for Indian Farmers")

query = st.text_input(
    "Ask your farming question",
    placeholder="गेहूं में कौन सा खाद डालें?"
)

if st.button("Ask"):

    if query:
        answer = answer_query(query)
        st.success(answer)
