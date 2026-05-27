from langchain_chroma import Chroma

from src.config.api_config import (
    PERSIST_DIRECTORY
)

def create_chroma_vectorstore(
    docs,
    embeddings
):

    return Chroma.from_documents(

        documents=docs,

        embedding=embeddings,

        persist_directory=PERSIST_DIRECTORY
    )

def load_chroma_vectorstore(
    embeddings
):

    return Chroma(

        persist_directory=PERSIST_DIRECTORY,

        embedding_function=embeddings
    )

def get_chroma_retriever(
    vectorstore
):

    return vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )