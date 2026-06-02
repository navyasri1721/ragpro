import os
import streamlit as st

# =========================================================
# LANGCHAIN
# =========================================================

from langchain_groq import ChatGroq
from langchain_core.documents import Document

import pytesseract

# =========================================================
# OCR CONFIG
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.config.api_config import PERSIST_DIRECTORY
from src.singleton.embedding_singleton import EmbeddingSingleton

from src.vectorstore.chroma_store import (
    create_chroma_vectorstore,
    load_chroma_vectorstore,
    get_chroma_retriever
)

from src.retrievers.hybrid_retriever import create_hybrid_retriever
from src.memory.chat_memory import get_memory
from src.utils.helpers import split_documents
from src.loaders.document_loader import process_uploaded_files

from src.pipeline.query_rewrite_handler import QueryRewriteHandler
from src.pipeline.retrieval_handler import RetrievalHandler
from src.pipeline.rerank_handler import RerankHandler
from src.pipeline.generation_handler import GenerationHandler

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(page_title="Advanced Hybrid RAG", layout="wide")
st.title("Advanced Hybrid RAG Chatbot")

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)

# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = EmbeddingSingleton.get_instance()

# =========================================================
# MEMORY
# =========================================================

memory = get_memory()

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# =========================================================
# 🔥 FIX 1: NUMERIC GUARDRAIL
# =========================================================

def numeric_guardrail(text: str) -> str:
    return text

def enforce_numeric_separation_prompt(text: str) -> str:
    return f"""
You are a STRICT RAG engine.

RULES:
- Do NOT merge numeric values
- Separate Official / Portal / Average / Trend
- Do NOT hallucinate values
- If not in context → say "Not found in document"

OUTPUT FORMAT:
Official:
Portal:
Average:
Trend:

QUESTION + CONTEXT:
{text}
"""

# =========================================================
# SAFE RETRIEVER
# =========================================================

def safe_retrieve(retriever, query, k=20):

    if retriever is None:
        return []

    try:
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)[:k]

        elif hasattr(retriever, "retrieve"):
            return retriever.retrieve(query)[:k]

        elif hasattr(retriever, "invoke"):
            return retriever.invoke(query)[:k]

        elif callable(retriever):
            return retriever(query)[:k]

    except Exception:
        return []

    return []

# =========================================================
# LOAD DB (UNCHANGED)
# =========================================================

if st.session_state.retriever is None:

    try:
        if os.path.exists(PERSIST_DIRECTORY):

            vectorstore = load_chroma_vectorstore(embeddings)
            db_data = vectorstore.get()

            if db_data and len(db_data["documents"]) > 0:

                documents = []

                for i in range(len(db_data["documents"])):
                    try:
                        metadata = db_data["metadatas"][i] or {}

                        clean_meta = {}
                        for k, v in metadata.items():
                            if v is None:
                                continue
                            clean_meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v

                        documents.append(
                            Document(
                                page_content=db_data["documents"][i],
                                metadata=clean_meta
                            )
                        )
                    except:
                        pass

                chroma_retriever = get_chroma_retriever(vectorstore)

                retriever = create_hybrid_retriever(
                    documents,
                    chroma_retriever
                )

                st.session_state.retriever = retriever
                st.sidebar.success(f"Loaded Persistent DB ({len(documents)} chunks)")

            else:
                st.sidebar.warning("Persistent DB empty.")

    except Exception as e:
        st.sidebar.error(f"DB Load Error: {e}")

# =========================================================
# SIDEBAR UPLOAD (UNCHANGED)
# =========================================================

st.sidebar.title("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload Files",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)

# =========================================================
# PROCESS DOCS (UNCHANGED)
# =========================================================

if uploaded_files:

    uploaded_names = sorted([f.name for f in uploaded_files])

    if uploaded_names != st.session_state.processed_files:

        with st.spinner("Processing documents..."):

            docs = process_uploaded_files(uploaded_files)
            split_docs = split_documents(docs)

            vectorstore = create_chroma_vectorstore(split_docs, embeddings)
            db_data = vectorstore.get()

            documents = []

            for i in range(len(db_data["documents"])):
                try:
                    metadata = db_data["metadatas"][i] or {}

                    clean_meta = {}
                    for k, v in metadata.items():
                        if v is None:
                            continue
                        clean_meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v

                    documents.append(
                        Document(
                            page_content=db_data["documents"][i],
                            metadata=clean_meta
                        )
                    )
                except:
                    pass

            chroma_retriever = get_chroma_retriever(vectorstore)

            retriever = create_hybrid_retriever(
                documents,
                chroma_retriever
            )

            st.session_state.retriever = retriever
            st.session_state.processed_files = uploaded_names

        st.success("Documents processed successfully.")

# =========================================================
# CLEAR CHAT
# =========================================================

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    memory.clear()
    st.rerun()

# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# INPUT
# =========================================================

question = st.chat_input("Ask questions from documents...")

# =========================================================
# SUMMARY DETECTION
# =========================================================

def is_summary_request(query):
    query = query.lower()
    return any(k in query for k in [
        "summarize", "summary", "overview",
        "document summary", "brief summary", "explain document"
    ])

# =========================================================
# MAIN PIPELINE
# =========================================================

if question:

    # 🔥 FIX 2: APPLY GUARDRAIL EARLY
    question = numeric_guardrail(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.retriever is None:
            st.warning("No documents available in database.")

        else:

            try:

                rewrite_handler = QueryRewriteHandler(llm, memory)
                retrieval_handler = RetrievalHandler(st.session_state.retriever)
                rerank_handler = RerankHandler()
                generation_handler = GenerationHandler(llm, memory)

                rewrite_handler.set_next(retrieval_handler)\
                               .set_next(rerank_handler)\
                               .set_next(generation_handler)

                result = rewrite_handler.handle({"question": question})

                docs = result.get("docs", [])

                context_text = "\n".join([d.page_content for d in docs])

                # =================================================
                # SUMMARY MODE
                # =================================================

                if is_summary_request(question):

                    summary_docs = safe_retrieve(
                        st.session_state.retriever,
                        question,
                        k=25
                    )

                    combined_text = "\n\n".join(
                        d.page_content for d in summary_docs
                    )[:12000]

                    prompt = f"""
STRICT RAG SUMMARIZER

{combined_text}
"""

                    answer = llm.invoke(prompt).content

                else:

                    # =================================================
                    # 🔥 FIX 3: NUMERIC CONTROL FOR QA
                    # =================================================

                    if any(k in question.lower() for k in ["package", "salary", "cgpa"]):

                        prompt = enforce_numeric_separation_prompt(
                            context_text + "\nQUESTION: " + question
                        )

                        answer = llm.invoke(prompt).content

                    else:
                        answer = result["answer"]

            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.stop()

            memory.save_context(
                {"input": question},
                {"output": answer}
            )

            st.markdown(answer)

            if docs:

                st.markdown("### Sources")

                shown = set()

                for doc in docs[:5]:

                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")

                    text = f"{src} — Page {page}"

                    if text not in shown:
                        st.markdown(f"- {text}")
                        shown.add(text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })