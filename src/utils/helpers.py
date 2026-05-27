from langchain_core.documents import Document
from src.preprocessing.chunker import TextChunker

def split_documents(docs):

    chunker = TextChunker()
    final_docs = []

    for d in docs:

        chunks = chunker.chunk_text(d["content"])

        for c in chunks:
            final_docs.append(
                Document(
                    page_content=c,
                    metadata={"source": d["source"]}
                )
            )

    return final_docs