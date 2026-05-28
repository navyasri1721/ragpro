from langchain_chroma import Chroma

from langchain_core.documents import (
    Document
)

from src.config.api_config import (
    PERSIST_DIRECTORY
)


# =========================================================
# CREATE VECTORSTORE
# =========================================================

def create_chroma_vectorstore(

    documents,

    embeddings
):

    vectorstore = Chroma(

        persist_directory=PERSIST_DIRECTORY,

        embedding_function=embeddings
    )

    # =====================================================
    # DELETE OLD FILES
    # =====================================================

    sources = set()

    for doc in documents:

        source = doc.metadata.get(
            "source"
        )

        if source:

            sources.add(source)

    for source in sources:

        try:

            existing = vectorstore.get(

                where={
                    "source": source
                }
            )

            ids = existing.get(
                "ids",
                []
            )

            if ids:

                vectorstore.delete(
                    ids=ids
                )

        except Exception as e:

            print(
                f"[DEBUG] DELETE ERROR: {e}"
            )

    # =====================================================
    # SAFE INSERT
    # =====================================================

    cleaned_docs = []

    for doc in documents:

        try:

            metadata = {}

            for k, v in doc.metadata.items():

                metadata[str(k)] = str(v)

            cleaned_docs.append(

                Document(

                    page_content=str(
                        doc.page_content
                    ),

                    metadata=metadata
                )
            )

        except Exception as e:

            print(
                f"[DEBUG] CLEAN ERROR: {e}"
            )

    print(
        f"[DEBUG] FINAL DOCS: {len(cleaned_docs)}"
    )

    vectorstore.add_documents(
        cleaned_docs
    )

    return vectorstore


# =========================================================
# LOAD VECTORSTORE
# =========================================================

def load_chroma_vectorstore(

    embeddings
):

    return Chroma(

        persist_directory=PERSIST_DIRECTORY,

        embedding_function=embeddings
    )


# =========================================================
# RETRIEVER
# =========================================================

def get_chroma_retriever(

    vectorstore
):

    return vectorstore.as_retriever(

        search_kwargs={
            "k": 25
        }
    )