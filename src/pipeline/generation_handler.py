from src.pipeline.base_handler import BaseHandler

class GenerationHandler(BaseHandler):

    def __init__(self, llm, memory):
        super().__init__()
        self.llm = llm
        self.memory = memory

    def handle(self, data):

        context = "\n".join([
            doc.page_content
            for doc in data["docs"]
        ])

        prompt = f"""
Context:
{context}

Question:
{data["question"]}

Answer:
"""

        response = self.llm.invoke(prompt)

        data["answer"] = response.content

        return super().handle(data)