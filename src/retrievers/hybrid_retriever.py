from langchain_community.retrievers import BM25Retriever
import streamlit as st

class HybridRetriever:

    def __init__(

        self,

        bm25_retriever,

        chroma_retriever
    ):

        self.bm25_retriever = bm25_retriever
        self.chroma_retriever = chroma_retriever

    def get_relevant_documents(

        self,

        query
    ):

        # =====================================
        # NORMAL RETRIEVAL
        # =====================================

        bm25_docs = (
            self.bm25_retriever.invoke(query)
        )

        chroma_docs = (
            self.chroma_retriever.invoke(query)
        )

        all_docs = bm25_docs + chroma_docs

        # =====================================
        # FILTER USING ACTIVE SOURCE
        # =====================================

        active_source = st.session_state.get(
            "active_source",
            None
        )

        if active_source:

            filtered_docs = []

            for doc in all_docs:

                source = doc.metadata.get(
                    "source",
                    ""
                )

                if active_source in source:

                    filtered_docs.append(doc)

            if filtered_docs:

                return filtered_docs[:5]

        return all_docs[:5]


def create_hybrid_retriever(

    documents,

    chroma_retriever
):

    bm25 = BM25Retriever.from_documents(
        documents
    )

    bm25.k = 4

    return HybridRetriever(

        bm25,

        chroma_retriever
    )