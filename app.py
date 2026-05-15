import streamlit as st
import os
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Kisan Saathi", page_icon="🌾")
st.title("🌾 Kisan Saathi")
st.write("Ask farming questions in Hindi or English")


# --------------------
# Gemini API
# --------------------
API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


# --------------------
# Embedding
# --------------------
class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return [
            genai.embed_content(
                model="models/embedding-001",
                content=t
            )["embedding"]
            for t in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(
            model="models/embedding-001",
            content=text
        )["embedding"]


embedding = GeminiEmbeddings()


# --------------------
# Load PDFs
# --------------------
docs = []

for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"docs/{file}")
        docs.extend(loader.load())


# --------------------
# Split docs
# --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


# --------------------
# Vector DB
# --------------------
db = FAISS.from_documents(
    chunks,
    embedding
)


# --------------------
# Ask
# --------------------
query = st.text_input("Ask question")


if st.button("Ask"):

    docs_found = db.similarity_search(
        query,
        k=3
    )

    context = "\n".join(
        [d.page_content for d in docs_found]
    )

    prompt = f"""
    You are an agriculture assistant.

    Context:
    {context}

    Farmer question:
    {query}

    Answer simply.
    Hindi if query is Hindi.
    """

    response = model.generate_content(prompt)

    st.success(response.text)
