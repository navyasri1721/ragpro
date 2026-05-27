from src.pipeline.base_handler import BaseHandler

class QueryRewriteHandler(BaseHandler):

    def __init__(self, llm, memory):
        super().__init__()
        self.llm = llm
        self.memory = memory

    def handle(self, data):
        data["question"] = data["question"].strip()
        return super().handle(data)