class QueryRewriteHandler:

    def __init__(self, llm, memory):

        self.llm = llm
        self.memory = memory

        # =====================================
        # IMPORTANT FIX
        # =====================================

        self.next_handler = None

    # =====================================
    # CHAINING
    # =====================================

    def set_next(self, handler):

        self.next_handler = handler

        return handler

    # =====================================
    # HANDLE
    # =====================================

    def handle(self, data):

        question = data.get(
            "question",
            ""
        )

        # =====================================
        # SIMPLE CLEANUP
        # =====================================

        rewritten_question = (
            question.strip()
            .lower()
        )

        data["question"] = rewritten_question

        # =====================================
        # NEXT HANDLER
        # =====================================

        if self.next_handler:

            return self.next_handler.handle(
                data
            )

        return data