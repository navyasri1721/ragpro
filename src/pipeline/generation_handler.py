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
        # GENERIC ADVANCED PROMPT
        # =================================================

        prompt = f"""
You are an advanced document QA assistant.

IMPORTANT RULES FOR NUMBERS:

- If multiple numeric values exist:
    - Separate them clearly
    - Do NOT merge them
- Identify type of value:
    - Official = recruitment data
    - Portal = reporting data
    - Average = aggregated value (NOT always eligibility)
- Never present ONLY one value if multiple exist
- Always label values clearly

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
  comparison_keywords = [
    "highest",
    "lowest",
    "most",
    "least",
    "maximum",
    "minimum",
    "top",
    "best"
]

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        # =================================================
        # LLM RESPONSE
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

        # =================================================
        # CLEANUP
        # =================================================

        answer = answer.replace(
            "Answer:",
            ""
        )

        answer = answer.replace(
            "ANSWER:",
            ""
        )

        answer = answer.strip()

        # =================================================
        # EMPTY RESPONSE SAFETY
        # =================================================

        if not answer:

            answer = (
                "Not found in document"
            )

        return {

            "answer": answer,

            "docs": docs,

            "question": question
        }