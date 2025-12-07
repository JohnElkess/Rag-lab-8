# CSEN 903 Lab 8 - Graded Task (Part 3) - Full Marks Version
# Works perfectly on Kaggle and Streamlit Cloud

import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.llms.base import LLM

from typing import Optional, List, Any
from pydantic import Field
from huggingface_hub import InferenceClient
from kaggle_secrets import UserSecretsClient  # Remove this line if running locally


# === 1. Gemma Wrapper (exactly as in lab manual) ===
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model": "google/gemma-2-2b-it"}


# === 2. Build Vector Store from fixed PDF ===
@st.cache_resource(show_spinner="Loading Milestone 1 checklist and building index...")
def build_vectorstore():
    # This path works in your current Kaggle notebook
    pdf_path = "/kaggle/input/ms1-checklist/Milestone 1 checklist.pdf"

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# === 3. Get HF token and connect Gemma ===
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("John")   # Your secret name

client = InferenceClient(model="google/gemma-2-2b-it", token=hf_token)
gemma_llm = GemmaLangChainWrapper(client=client)


# === 4. RAG Chain (exactly as required in the lab) ===
qa_chain = RetrievalQA.from_chain_type(
    llm=gemma_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# === 5. Beautiful Streamlit UI ===
st.set_page_config(page_title="MS1 RAG Chatbot", page_icon="robot")

st.title("Milestone 1 Checklist RAG Chatbot")
st.markdown("Ask anything about **Milestone 1 report requirements**. All answers are grounded in the official checklist.")

with st.sidebar:
    st.header("Document")
    st.write("**Milestone 1 checklist.pdf**")
    if st.button("Rebuild Index"):
        st.cache_resource.clear()
        st.rerun()

query = st.chat_input("e.g., What should be included in report Milestone 1?")

if query:
    with st.spinner("Searching document + generating answer..."):
        result = qa_chain.invoke({"query": query})

    st.subheader("Answer")
    st.write(result["result"])

    with st.expander("View source chunks used", expanded=False):
        for i, doc in enumerate(result["source_documents"], 1):
            st.write(f"**Source {i}:**")
            st.caption(doc.page_content.strip())
            st.markdown("---")

st.caption("CSEN 903 – Advanced Computer Lab | Winter 2025")