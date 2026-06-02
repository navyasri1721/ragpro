from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config.api_config import PERSIST_DIRECTORY

# =========================================================
# SAFE METADATA CLEANER (FINAL FIX)
# =========================================================

def safe_value(value):
    """
    Chroma ONLY accepts:
    str | int | float | bool
    """

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    # convert lists/dicts safely
    try:
        return str(value)
    except:
        return ""


def clean_metadata(metadata):
    cleaned = {}

    if not metadata:
        return cleaned

    for k, v in metadata.items():
        try:
            key = str(k)
            cleaned[key] = safe_value(v)
        except:
            continue

    return cleaned


# =========================================================
# CREATE / UPDATE VECTORSTORE (FIXED)
# =========================================================

def create_chroma_vectorstore(documents, embeddings):

    cleaned_docs = []

    # ===============================
    # CLEAN DOCUMENTS SAFELY
    # ===============================
    for idx, doc in enumerate(documents):

        try:
            content = getattr(doc, "page_content", "")

            if not content:
                continue

            content = str(content).strip()

            metadata = clean_metadata(getattr(doc, "metadata", {}))

            # SAFE DEFAULT FIELDS
            metadata.update({
                "source": str(metadata.get("source", "unknown")),
                "page": str(metadata.get("page", "N/A")),
                "type": str(metadata.get("type", "text")),
                "chunk_id": str(idx)
            })

            cleaned_docs.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        except Exception as e:
            print(f"[DOC CLEAN ERROR {idx}] {e}")

    if not cleaned_docs:
        raise ValueError("No valid documents after cleaning.")

    print(f"[DEBUG] CLEAN DOCS: {len(cleaned_docs)}")

    # ===============================
    # INIT VECTORSTORE
    # ===============================
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # ===============================
    # SAFE BATCH INSERT (FIX CRASH)
    # ===============================
    BATCH_SIZE = 32

    for i in range(0, len(cleaned_docs), BATCH_SIZE):

        batch = cleaned_docs[i:i + BATCH_SIZE]

        try:
            vectorstore.add_documents(batch)
        except Exception as e:

            print(f"[BATCH ERROR {i}] {e}")

            # fallback: insert one-by-one
            for j, doc in enumerate(batch):
                try:
                    vectorstore.add_documents([doc])
                except Exception as e2:
                    print(f"[SKIP DOC {i+j}] {e2}")
                    print("METADATA:", doc.metadata)

    print("[DEBUG] VECTORSTORE READY")

    return vectorstore


# =========================================================
# LOAD VECTORSTORE (FIXED)
# =========================================================

def load_chroma_vectorstore(embeddings):

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )


# =========================================================
# RETRIEVER (FIXED SAFE DEFAULT)
# =========================================================

def get_chroma_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )