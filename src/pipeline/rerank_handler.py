from src.pipeline.base_handler import BaseHandler

from src.reranker.reranker import (
    rerank_documents
)


# =========================================================
# RERANK HANDLER
# =========================================================

class RerankHandler(BaseHandler):

    def handle(self, data):

        docs = data.get("docs", [])

        query = data.get(

            "rewritten_query",

            data.get("question", "")
        )

        if not docs:

            data["docs"] = []

            return super().handle(data)

        # =================================================
        # CROSS ENCODER RERANKING
        # =================================================

        reranked_docs = rerank_documents(

            query,

            docs,

            top_k=6
        )

        data["docs"] = reranked_docs

        return super().handle(data)