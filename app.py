import streamlit as st
import os
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Kisan Saathi",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Kisan Saathi")
st.subheader("AI Agriculture Assistant for Indian Farmers")


# -----------------------------
# Gemini API Key
# -----------------------------
API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


# -----------------------------
# Gemini Embeddings
# -----------------------------
class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text
            )
            embeddings.append(response["embedding"])

        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )

        return response["embedding"]


embedding = GeminiEmbeddings()


# -----------------------------
# Load PDF Documents
# -----------------------------
def load_documents():

    docs = []

    if not os.path.exists("docs"):
        st.error("docs folder missing")
        st.stop()

    for file in os.listdir("docs"):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(f"docs/{file}")
            docs.extend(loader.load())

    return docs


documents = load_documents()


# -----------------------------
# Split Documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)


# -----------------------------
# Create Vector Database
# -----------------------------
db = FAISS.from_documents(
    chunks,
    embedding
)


# -----------------------------
# Query Function
# -----------------------------
def answer_query(query):

    docs = db.similarity_search(query, k=3)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are Kisan Saathi.

    Answer farmer questions simply.

    Use this agriculture advisory context:

    {context}

    Farmer Question:
    {query}

    Rules:
    - Give practical answer
    - Keep answer short
    - Hindi if question is Hindi
    - Mention fertilizer/pest/irrigation clearly
    """

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# User Input
# -----------------------------
query = st.text_input(
    "Ask your farming question",
    placeholder="गेहूं में यूरिया कब डालें?"
)

if st.button("Ask"):

    if query.strip():

        with st.spinner("Thinking..."):
            answer = answer_query(query)

        st.success(answer)

    else:
        st.warning("Please enter question")
