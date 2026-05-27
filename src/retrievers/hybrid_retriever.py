from langchain.retrievers import BM25Retriever, EnsembleRetriever

def create_hybrid_retriever(docs, chroma_retriever):

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5

    return EnsembleRetriever(
        retrievers=[bm25, chroma_retriever],
        weights=[0.5, 0.5]
    )