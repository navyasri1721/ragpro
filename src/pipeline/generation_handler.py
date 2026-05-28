class GenerationHandler:

    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.next_handler = None

    # =========================================
    # CHAIN SETUP
    # =========================================
    def set_next(self, handler):
        self.next_handler = handler
        return handler

    # =========================================
    # MAIN HANDLE FUNCTION
    # =========================================
    def handle(self, data):

        question = data.get("question", "")
        docs = data.get("docs", [])

        # If next handler exists, process chain first
        if self.next_handler:
            result = self.next_handler.handle(data)
            docs = result.get("docs", docs)
            question = result.get("question", question)

        # =========================================
        # BUILD CONTEXT (FROM RETRIEVED DOCS)
        # =========================================
        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        # =========================================
        # 🔥 STRICT PROMPT (IMPORTANT FIX)
        # =========================================
        prompt = f"""
You are a strict document-based QA assistant.

RULES:
- Use ONLY the given context
- Do NOT use external knowledge
- If answer is not in context, reply exactly: Not found in document
- Do NOT explain anything
- Do NOT add extra text, reasoning, or assumptions
- Give ONLY the final answer (short and direct)

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # =========================================
        # LLM CALL
        # =========================================
        response = self.llm.invoke(prompt)

        answer = response.content if hasattr(response, "content") else str(response)

        # =========================================
        # FINAL SAFETY CLEANUP
        # =========================================
        answer = answer.strip()

        if "\n" in answer:
            answer = answer.split("\n")[0]

        # =========================================
        # RETURN RESULT
        # =========================================
        return {
            "answer": answer,
            "docs": docs,
            "question": question
        }