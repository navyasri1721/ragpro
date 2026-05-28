from langchain_chroma import Chroma
from src.config.api_config import PERSIST_DIRECTORY

# =====================================================
# CREATE CHROMA VECTORSTORE (FIXED + SAFE)
# =====================================================

def create_chroma_vectorstore(documents, embeddings):

    # =================================================
    # STEP 1: CLEAN + VALIDATE DOCUMENTS
    # =================================================

    cleaned_docs = []

    for doc in documents:

        # ensure page_content exists
        if not hasattr(doc, "page_content"):
            continue

        content = doc.page_content

        if content and content.strip():

            # ensure metadata exists
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}

            # enforce source tracking
            doc.metadata["source"] = doc.metadata.get(
                "source",
                "unknown"
            )

            cleaned_docs.append(doc)

    # =================================================
    # STEP 2: VALIDATION CHECK (IMPORTANT)
    # =================================================

    if len(cleaned_docs) == 0:

        raise ValueError(
            "No valid documents found for vectorstore creation."
        )

    # =================================================
    # STEP 3: CREATE VECTORSTORE
    # =================================================

    vectorstore = Chroma.from_documents(
        documents=cleaned_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore


# =====================================================
# LOAD EXISTING VECTORSTORE
# =====================================================

def load_chroma_vectorstore(embeddings):

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )


# =====================================================
# RETRIEVER WRAPPER
# =====================================================

def get_chroma_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )