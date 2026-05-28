from langchain_community.retrievers import BM25Retriever


# =========================================================
# HYBRID RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(

        self,

        bm25_retriever,

        chroma_retriever
    ):

        self.bm25_retriever = bm25_retriever

        self.chroma_retriever = chroma_retriever

    # =====================================================
    # MAIN RETRIEVAL
    # =====================================================

    def get_relevant_documents(

        self,

        query
    ):

        # =================================================
        # BM25 RETRIEVAL
        # =================================================

        bm25_docs = self.bm25_retriever.invoke(
            query
        )

        # =================================================
        # VECTOR RETRIEVAL
        # =================================================

        chroma_docs = self.chroma_retriever.invoke(
            query
        )

        # =================================================
        # MERGE
        # =================================================

        all_docs = bm25_docs + chroma_docs

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        unique_docs = []

        seen = set()

        for doc in all_docs:

            content_key = (
                doc.page_content[:300]
            )

            if content_key not in seen:

                unique_docs.append(doc)

                seen.add(content_key)

        return unique_docs[:10]


# =========================================================
# CREATE HYBRID RETRIEVER
# =========================================================

def create_hybrid_retriever(

    documents,

    chroma_retriever
):

    bm25 = BM25Retriever.from_documents(
        documents
    )

    bm25.k = 8

    return HybridRetriever(

        bm25,

        chroma_retriever
    )