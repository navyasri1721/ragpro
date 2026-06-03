import os
import streamlit as st

# =========================================================
# LANGCHAIN
# =========================================================

from langchain_groq import ChatGroq

from langchain_core.documents import (
    Document
)

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

from src.config.api_config import (
    PERSIST_DIRECTORY
)

from src.singleton.embedding_singleton import (
    EmbeddingSingleton
)

from src.vectorstore.chroma_store import (

    create_chroma_vectorstore,

    load_chroma_vectorstore,

    get_chroma_retriever
)

from src.retrievers.hybrid_retriever import (
    create_hybrid_retriever
)

from src.memory.chat_memory import (
    get_memory
)

from src.utils.helpers import (
    split_documents
)

from src.loaders.document_loader import (
    process_uploaded_files
)

from src.pipeline.query_rewrite_handler import (
    QueryRewriteHandler
)

from src.pipeline.retrieval_handler import (
    RetrievalHandler
)

from src.pipeline.rerank_handler import (
    RerankHandler
)

from src.pipeline.generation_handler import (
    GenerationHandler
)
from src.utils.source_utils import group_by_source, pick_best_source

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(

    page_title="Advanced Hybrid RAG",

    layout="wide"
)

st.title(
    "Advanced Hybrid RAG Chatbot"
)

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(

    groq_api_key=st.secrets[
        "GROQ_API_KEY"
    ],

    model_name="llama-3.3-70b-versatile"
)

# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = (
    EmbeddingSingleton.get_instance()
)

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
# AUTO LOAD PERSISTENT CHROMADB
# =========================================================

if st.session_state.retriever is None:

    try:

        if os.path.exists(PERSIST_DIRECTORY):

            vectorstore = load_chroma_vectorstore(
                embeddings
            )

            # =====================================
            # CHECK IF DB HAS DATA
            # =====================================

            db_data = vectorstore.get()

            if (

                db_data

                and

                len(
                    db_data["documents"]
                ) > 0
            ):

                documents = []

                for i in range(

                    len(
                        db_data["documents"]
                    )
                ):

                    try:

                        doc = Document(

                            page_content=db_data[
                                "documents"
                            ][i],

                            metadata=db_data[
                                "metadatas"
                            ][i]
                        )

                        documents.append(
                            doc
                        )

                    except:
                        pass

                chroma_retriever = (
                    get_chroma_retriever(
                        vectorstore
                    )
                )

                retriever = (
                    create_hybrid_retriever(

                        documents,

                        chroma_retriever
                    )
                )

                st.session_state.retriever = (
                    retriever
                )

                st.sidebar.success(
                    f"Loaded Persistent DB ({len(documents)} chunks)"
                )

            else:

                st.sidebar.warning(
                    "Persistent DB empty."
                )

    except Exception as e:

        st.sidebar.error(
            f"DB Load Error: {e}"
        )

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Upload Documents"
)

st.sidebar.write(
    "Supported Formats:"
)

st.sidebar.write(
    "PDF | DOCX | TXT | CSV"
)

uploaded_files = st.sidebar.file_uploader(

    "Upload Files",

    type=[
        "pdf",
        "docx",
        "txt",
        "csv"
    ],

    accept_multiple_files=True
)

# =========================================================
# PROCESS DOCUMENTS
# =========================================================

if uploaded_files:

    uploaded_names = sorted([

        file.name

        for file in uploaded_files
    ])

    if (

        uploaded_names

        != st.session_state.processed_files
    ):

        with st.spinner(
            "Processing documents..."
        ):

            # =====================================
            # LOAD
            # =====================================

            docs = process_uploaded_files(
                uploaded_files
            )

            print(
                f"[DEBUG] RAW DOCS: {len(docs)}"
            )

            # =====================================
            # SPLIT
            # =====================================

            split_docs = split_documents(
                docs
            )

            print(
                f"[DEBUG] SPLIT DOCS: {len(split_docs)}"
            )

            # =====================================
            # CREATE / UPDATE VECTORSTORE
            # =====================================

            vectorstore = (
                create_chroma_vectorstore(

                    split_docs,

                    embeddings
                )
            )

            # =====================================
            # LOAD ALL DOCS FROM DB
            # =====================================

            db_data = vectorstore.get()

            documents = []

            for i in range(

                len(
                    db_data["documents"]
                )
            ):

                try:

                    documents.append(

                        Document(

                            page_content=db_data[
                                "documents"
                            ][i],

                            metadata=db_data[
                                "metadatas"
                            ][i]
                        )
                    )

                except:
                    pass

            # =====================================
            # RETRIEVER
            # =====================================

            chroma_retriever = (
                get_chroma_retriever(
                    vectorstore
                )
            )

            retriever = (
                create_hybrid_retriever(

                    documents,

                    chroma_retriever
                )
            )

            st.session_state.retriever = (
                retriever
            )

            st.session_state.processed_files = (
                uploaded_names
            )

        st.success(
            "Documents processed successfully."
        )

# =========================================================
# CLEAR CHAT
# =========================================================

if st.sidebar.button(
    "Clear Chat"
):

    st.session_state.messages = []

    memory.clear()

    st.rerun()

# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(
    "Ask questions from documents..."
)

# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    st.session_state.messages.append({

        "role": "user",

        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.retriever is None:

            st.warning(
                "No documents available in database."
            )

        else:

            try:

                # =================================
                # HANDLERS
                # =================================

                rewrite_handler = (
                    QueryRewriteHandler(
                        llm,
                        memory
                    )
                )

                retrieval_handler = (
                    RetrievalHandler(
                        st.session_state.retriever
                    )
                )

                rerank_handler = (
                    RerankHandler()
                )

                generation_handler = (
                    GenerationHandler(
                        llm,
                        memory
                    )
                )

                # =================================
                # CHAIN
                # =================================

                rewrite_handler.set_next(
                    retrieval_handler
                ).set_next(
                    rerank_handler
                ).set_next(
                    generation_handler
                )

                # =================================
                # RUN
                # =================================

                result = rewrite_handler.handle({

                    "question": question
                })

                docs = result.get("docs", [])

                answer = "Not found in document"
                context_text = ""

                try:
                    if not docs:
                        answer = "Not found in document"
                    else:
                        grouped = group_by_source(docs)
                        docs = pick_best_source(grouped, question)

                        context_text = "\n".join(
            d.page_content for d in docs if d.page_content
        )

                        if not context_text.strip():
                            answer = "Not found in document"
                        else:
                            prompt = f"""
You are a STRICT RAG SYSTEM.

RULES:
- Answer ONLY from context
- Do NOT mix sources
- Do NOT guess
- If not in context → say "Not found in document"

CONTEXT:
{context_text}

QUESTION:
{question}
"""

                            answer = llm.invoke(prompt).content
                except Exception as e:
                        answer = f"Error generating answer: {str(e)}"
            except Exception as e:
                    
                st.error(
                    f"Pipeline Error: {e}"
                )

                st.stop()

            # =================================
            # SAVE MEMORY
            # =================================

            memory.save_context(

                {"input": question},

                {"output": answer}
            )

            # =================================
            # SHOW ANSWER
            # =================================

            st.markdown(answer)

            # =================================
            # SOURCES
            # =================================

            if docs:

                st.markdown(
                    "### Sources"
                )

                shown_sources = set()

                for doc in docs[:5]:

                    source = doc.metadata.get(

                        "source",

                        "Unknown"
                    )

                    page = doc.metadata.get(

                        "page",

                        "N/A"
                    )

                    source_text = (
                        f"{source} — Page {page}"
                    )

                    if (

                        source_text

                        not in shown_sources
                    ):

                        st.markdown(
                            f"- {source_text}"
                        )

                        shown_sources.add(
                            source_text
                        )

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer
    })