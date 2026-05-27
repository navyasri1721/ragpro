from src.pipeline.base_handler import BaseHandler

class RerankHandler(BaseHandler):

    def handle(self, data):

        docs = data["docs"]

        data["docs"] = docs[:5]

        return super().handle(data)