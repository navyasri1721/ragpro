import os
import streamlit as st
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import tempfile
from db.mysql_search import search_mysql

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
from src.web.web_search import search_web
from db.chat_storage import save_chat
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
groq_client = Groq(

    api_key=st.secrets[
        "GROQ_API_KEY"
    ]
)
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
# =========================================================
# SELECT SPEAKING LANGUAGE
# =========================================================

selected_language = st.selectbox(

    "Select Speaking Language",

    [
        "English",
        "Telugu",
        "Hindi",
        "Tamil",
        "Kannada",
        "Malayalam"
    ]
)

language_map = {

    "English": "en",

    "Telugu": "te",

    "Hindi": "hi",

    "Tamil": "ta",

    "Kannada": "kn",

    "Malayalam": "ml"
}

selected_language_code = language_map[selected_language]
st.write(f"Selected Language: {selected_language}")
# =========================================================
# VOICE INPUT
# =========================================================

voice_data = mic_recorder(

    start_prompt="🎤 Start Recording",

    stop_prompt="⏹ Stop Recording",

    key="voice_recorder"
)

voice_question = None

if voice_data:

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(
            voice_data["bytes"]
        )

        temp_audio_path = (
            temp_audio.name
        )

    with open(
        temp_audio_path,
        "rb"
    ) as audio_file:

        transcription = (
    groq_client.audio.transcriptions.create(

        file=audio_file,

        model="whisper-large-v3",

        language=selected_language_code
    )
)

    voice_question = (
        transcription.text
    )
    translation_prompt = f"""
Translate the following user query into simple English.
Preserve meaning exactly.

TEXT:
{voice_question}

Only return translated English text.
"""
    if selected_language != "English":
        translated_question = llm.invoke(
        translation_prompt
    ).content.strip()
        voice_question = translated_question
    st.info(
    f"Detected Voice Text: {transcription.text}"
)

    st.success(
        f"Voice Input: {voice_question}"
    )
typed_question = st.chat_input(
    "Ask questions from documents..."
)

question = (

    typed_question

    if typed_question

    else voice_question
)
web_keywords = [

    "weather",
    "ceo",
    "today",
    "news",
    "current",
    "latest",
    "stock",
    "price",
    "share",
    "market",
    "president",
    "prime minister",
    "founder",
    "owner"
]

force_web = False

if question:

    force_web = any(

        word in question.lower()

        for word in web_keywords
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

                docs = []

# =================================
# RUN RAG ONLY IF RETRIEVER EXISTS
# =================================

                if st.session_state.retriever is not None:
                    result = rewrite_handler.handle({
                        "question": question
                        })
                    docs = result.get(
        "docs",
        []
    )

                answer = "Not found in document"

                context_text = ""

                # =====================================
                # CHECK DOCUMENT RELEVANCE
                # =====================================

                relevant_docs = []
                if docs:
                    stop_words = {
    "what",
    "who",
    "is",
    "the",
    "of",
    "does",
    "today",
    "current",
    "tell",
    "about",
    "explain",
    "which",
    "company"
}
                    question_words = set(
            re.findall(r'\w+', question.lower())
        )    
                    filtered_words = {
        word
        for word in question_words
        if word not in stop_words and len(word) > 2
    }
                    for doc in docs:

                        content = (
                            doc.page_content.strip()
                        )

                        # Ignore tiny chunks

                        if len(content) < 40:
                            continue

                        content_words = set(
            re.findall(r'\w+', content.lower())
        )
       
                        overlap = len(
            filtered_words.intersection(content_words)
        )
                        if overlap >= 1:
                            relevant_docs.append(doc)     

                if force_web:
                    relevant_docs = []
            
                                               # =====================================
                # MYSQL DATABASE SEARCH
                # =====================================

                mysql_results = []

                if not relevant_docs:

                    mysql_results = search_mysql(question)

                    if mysql_results:

                        row = mysql_results[0]

                        answer_parts = []

                        lower_question = question.lower()

                        # -----------------------------
                        # CEO QUESTIONS
                        # -----------------------------
                        if "ceo" in lower_question:

                            ceo = row.get("ceo")

                            if ceo:

                                answer_parts.append(
                                    f"CEO: {ceo}"
                                )

                        # -----------------------------
                        # PACKAGE QUESTIONS
                        # -----------------------------
                        elif "package" in lower_question:

                            package = row.get("package_lpa")

                            if package:

                                answer_parts.append(
                                    f"Package: {package} LPA"
                                )

                        # -----------------------------
                        # CGPA QUESTIONS
                        # -----------------------------
                        elif "cgpa" in lower_question:

                            cgpa = row.get("cgpa_required")

                            if cgpa:

                                answer_parts.append(
                                    f"CGPA Required: {cgpa}"
                                )

                        # -----------------------------
                        # INTERVIEW QUESTIONS
                        # -----------------------------
                        elif "interview" in lower_question:

                            focus = row.get("interview_focus")

                            if focus:

                                answer_parts.append(
                                    f"Interview Focus: {focus}"
                                )

                        # -----------------------------
                        # DEFAULT COMPANY INFO
                        # -----------------------------
                        else:

                            company = row.get("company_name")

                            if company:

                                answer_parts.append(
                                    f"Company: {company}"
                                )

                        # -----------------------------
                        # FINAL ANSWER
                        # -----------------------------
                        if answer_parts:

                            answer = "\n".join(answer_parts)

                            st.markdown(
                                "### Source Type"
                            )

                            st.write(
                                "🗄️ MySQL Database"
                            )

                            docs = []

                        else:

                            mysql_results = []
                # =====================================
                # WEB SEARCH FALLBACK
                # =====================================

                if not relevant_docs and not mysql_results:
                    web_results = search_web(question)
                    
                    print("WEB RESULTS:")
                    print(web_results)

                    if not web_results or len(web_results) == 0:
                        answer = "Could not find reliable web results."

                    else:

                        web_context = "\n\n".join([

                            f"Title: {r['title']}\n"
                            f"Content: {r['body']}"

                            for r in web_results
                        ])

                        prompt = f"""
You are a helpful AI assistant.

Use the WEB RESULTS below to answer the QUESTION.

Rules:
- Extract exact factual answer from results.
- Keep answer short and direct.
- If asking about a CEO, return the CEO name clearly.
- Do not hallucinate.
- If answer missing, say:
  "Not found on web."

WEB RESULTS:
{web_context}

QUESTION:
{question}

ANSWER:
"""

                        answer = (
                            llm.invoke(
                                prompt
                            ).content.strip()
                        )

                        st.markdown(
                            "### Source Type"
                        )

                        st.write(
                            "🌐 Web Search"
                        )

                        docs = []

                # =====================================
                # DOCUMENT RAG
                # =====================================

                elif relevant_docs:

                    grouped = group_by_source(
                        relevant_docs
                    )

                    docs = pick_best_source(

                        grouped,

                        question
                    )

                    context_text = "\n".join(

                        d.page_content

                        for d in docs

                        if d.page_content
                    )

                    if not context_text.strip():

                        answer = (
                            "Not found in document"
                        )

                    else:

                        prompt = f"""
You are a helpful RAG assistant.

RULES:
- Answer ONLY using the document context.
- If the answer is partially available, summarize it clearly.
- If exact wording is not present but meaning is clear, answer naturally.
- Do NOT invent facts outside context.
- If answer truly does not exist, say:
  "Not found in document"

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""

                        response = (
                            llm.invoke(
                                prompt
                            ).content.strip()
                        )

                        if (

                            not response

                            or

                            "not found"

                            in response.lower()
                        ):

                            answer = (
                                "Not found in document"
                            )

                        else:

                            answer = response

                        st.markdown(
                            "### Source Type"
                        )

                        st.write(
                            "📄 Document RAG"
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

                st.markdown(
                    answer
                )

                # =================================
                # SAVE CHAT TO MYSQL
                # =================================

                save_chat(
                    question,
                    answer
                )

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

                # =================================
                # SAVE TO SESSION
                # =================================

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"Pipeline Error: {e}"
                )

                st.stop()