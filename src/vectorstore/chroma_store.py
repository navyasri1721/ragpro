from langchain_chroma import Chroma
from src.config import PERSIST_DIRECTORY

def create_vectorstore(docs, embeddings):

    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )