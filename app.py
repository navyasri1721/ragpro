import os
import streamlit as st

# =========================================================
# LANGCHAIN
# =========================================================

from langchain_groq import ChatGroq

from langchain_core.documents import (
    Document
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
# LOAD EXISTING CHROMADB
# =========================================================

if (

    os.path.exists(PERSIST_DIRECTORY)

    and st.session_state.retriever is None
):

    try:

        vectorstore = (
            load_chroma_vectorstore(
                embeddings
            )
        )

        all_docs = vectorstore.get()

        documents = []

        if (

            all_docs

            and all_docs["documents"]
        ):

            for i in range(

                len(
                    all_docs["documents"]
                )
            ):

                doc = Document(

                    page_content=all_docs[
                        "documents"
                    ][i],

                    metadata=all_docs[
                        "metadatas"
                    ][i]
                )

                documents.append(doc)

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
            "Persistent ChromaDB Loaded"
        )

    except Exception as e:

        st.sidebar.error(
            f"Database Load Error: {e}"
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
            "Processing Documents..."
        ):

            # =====================================
            # LOAD DOCUMENTS
            # =====================================

            docs = process_uploaded_files(
                uploaded_files
            )

            # =====================================
            # SPLIT DOCUMENTS
            # =====================================

            split_docs = split_documents(
                docs
            )

            # =====================================
            # CREATE VECTORSTORE
            # =====================================

            vectorstore = (
                create_chroma_vectorstore(

                    split_docs,

                    embeddings
                )
            )

            # =====================================
            # CREATE RETRIEVER
            # =====================================

            chroma_retriever = (
                get_chroma_retriever(
                    vectorstore
                )
            )

            retriever = (
                create_hybrid_retriever(

                    split_docs,

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
            "Documents Processed Successfully"
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
# DISPLAY CHAT HISTORY
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
    "Ask questions from uploaded documents..."
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
                "Please upload documents first."
            )

        else:

            try:

                # =================================
                # CREATE HANDLERS
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
                # CHAIN OF RESPONSIBILITY
                # =================================

                rewrite_handler.set_next(
                    retrieval_handler
                ).set_next(
                    rerank_handler
                ).set_next(
                    generation_handler
                )

                # =================================
                # RUN PIPELINE
                # =================================

                result = rewrite_handler.handle({

                    "question": question
                })

                answer = result["answer"]

                docs = result.get(
                    "docs",
                    []
                )

            except Exception as e:

                st.error(
                    f"Pipeline Error: {e}"
                )

                st.stop()

            # =================================
            # EMPTY RESPONSE
            # =================================

            if not docs:

                answer = (
                    "No relevant information found."
                )

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
            # SHOW SOURCES
            # =================================

            if docs:

                st.markdown(
                    "### Sources"
                )

                shown_sources = set()

                for doc in docs[:3]:

                    source = doc.metadata.get(

                        "source",

                        "Unknown File"
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