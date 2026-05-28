import streamlit as st

from src.pipeline.base_handler import BaseHandler


class RetrievalHandler(BaseHandler):

    def __init__(self, retriever):

        super().__init__()

        self.retriever = retriever

    def handle(self, data):

        # =====================================
        # USE REWRITTEN QUERY
        # =====================================

        query = data.get(

            "rewritten_query",

            data["question"]

        )

        # =====================================
        # RETRIEVE DOCUMENTS
        # =====================================

        docs = (

            self.retriever.get_relevant_documents(

                query

            )

        )

        # =====================================
        # STORE ACTIVE SOURCE
        # =====================================

        if docs:

            st.session_state.active_source = (

                docs[0].metadata.get(

                    "source",

                    ""

                )

            )

        # =====================================
        # SAVE DOCS
        # =====================================

        data["docs"] = docs

        return super().handle(data)