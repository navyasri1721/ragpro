from src.pipeline.base_handler import BaseHandler

class RetrievalHandler(BaseHandler):

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever

    def handle(self, data):

        docs = self.retriever.get_relevant_documents(
            data["question"]
        )

        data["docs"] = docs

        return super().handle(data)