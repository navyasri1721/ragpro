class GenerationHandler:

    def __init__(

        self,

        llm,

        memory
    ):

        self.llm = llm

        self.memory = memory

        self.next_handler = None

    # =====================================================
    # CHAINING
    # =====================================================

    def set_next(

        self,

        handler
    ):

        self.next_handler = handler

        return handler

    # =====================================================
    # HANDLE
    # =====================================================

    def handle(

        self,

        data
    ):

        question = data.get(
            "question",
            ""
        )

        docs = data.get(
            "docs",
            []
        )

        # =================================================
        # BUILD CONTEXT
        # =================================================

        context = "\n\n".join(

            [

                doc.page_content

                for doc in docs
            ]
        )

        # =================================================
        # IMPROVED PROMPT
        # =================================================

        prompt = f"""
You are an advanced document QA assistant.

You MUST answer ONLY from the provided context.

RULES:
- Use retrieved context carefully
- Tables may contain the answer
- Infer simple table relationships if obvious
- For example:
    - bond = 0 means bond-free
    - backlog = 0 means no backlogs allowed
- Answer comparison questions carefully
- Answer ranking/filtering questions carefully
- Keep answers concise
- If truly absent, say:
  Not found in document

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # =================================================
        # LLM
        # =================================================

        response = self.llm.invoke(
            prompt
        )

        answer = (

            response.content

            if hasattr(
                response,
                "content"
            )

            else str(response)
        )

        answer = answer.strip()

        return {

            "answer": answer,

            "docs": docs,

            "question": question
        }