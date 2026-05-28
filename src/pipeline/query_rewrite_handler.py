class QueryRewriteHandler:

    def __init__(

        self,

        llm,

        memory
    ):

        self.llm = llm

        self.memory = memory

        self.next_handler = None

    # =====================================================
    # SET NEXT
    # =====================================================

    def set_next(self, handler):

        self.next_handler = handler

        return handler

    # =====================================================
    # HANDLE
    # =====================================================

    def handle(self, data):

        question = data.get(
            "question",
            ""
        )

        rewritten_question = (
            question.strip()
        )

        # IMPORTANT FIX
        data["rewritten_query"] = (
            rewritten_question
        )

        # =================================================
        # NEXT HANDLER
        # =================================================

        if self.next_handler:

            return self.next_handler.handle(
                data
            )

        return data