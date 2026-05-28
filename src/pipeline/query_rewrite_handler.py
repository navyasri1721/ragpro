import streamlit as st

from src.pipeline.base_handler import BaseHandler


class QueryRewriteHandler(BaseHandler):

    def __init__(self, llm, memory):

        super().__init__()

        self.llm = llm
        self.memory = memory

        # =====================================
        # SESSION STATE
        # =====================================

        if "last_full_question" not in st.session_state:

            st.session_state.last_full_question = ""

    def handle(self, data):

        question = data["question"]

        lower_q = question.lower()

        # =====================================
        # FOLLOW-UP DETECTION
        # =====================================

        followup_words = [

            "its",
            "their",
            "those",
            "types",
            "advantages",
            "disadvantages"

        ]

        is_followup = any(

            word in lower_q

            for word in followup_words

        )

        # =====================================
        # REWRITE FOLLOW-UP
        # =====================================

        if (

            is_followup

            and st.session_state.last_full_question

        ):

            rewritten_query = (

                f"{question} of "
                f"{st.session_state.last_full_question}"

            )

        else:

            rewritten_query = question

            # store latest complete topic
            st.session_state.last_full_question = question

        # =====================================
        # SAVE REWRITTEN QUERY
        # =====================================

        data["rewritten_query"] = rewritten_query

        return super().handle(data)