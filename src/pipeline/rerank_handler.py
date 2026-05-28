from src.pipeline.base_handler import BaseHandler

# =====================================================
# IMPROVED RERANK HANDLER
# =====================================================

class RerankHandler(BaseHandler):

    def handle(self, data):

        docs = data.get("docs", [])
        query = data.get("rewritten_query", "").lower()

        if not docs:

            data["docs"] = []

            return super().handle(data)

        # =================================================
        # STEP 1: SCORE DOCUMENTS
        # =================================================

        scored_docs = []

        for doc in docs:

            content = doc.page_content.lower()

            source = doc.metadata.get("source", "unknown")

            # -----------------------------------------
            # scoring logic (simple but effective)
            # -----------------------------------------

            score = 0

            # keyword match boost
            if query in content:
                score += 3

            # partial match boost
            query_words = query.split()

            for word in query_words:

                if word in content:
                    score += 1

            # smaller penalty for duplicates
            score -= len(content) * 0.0001

            scored_docs.append((score, doc))

        # =================================================
        # STEP 2: SORT BY SCORE
        # =================================================

        scored_docs.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # =================================================
        # STEP 3: SOURCE DIVERSITY (IMPORTANT FIX)
        # =================================================

        seen_sources = set()

        final_docs = []

        for score, doc in scored_docs:

            source = doc.metadata.get("source", "unknown")

            # ensure diversity across documents
            if source in seen_sources:
                continue

            seen_sources.add(source)

            final_docs.append(doc)

            if len(final_docs) == 5:
                break

        # =================================================
        # FINAL OUTPUT
        # =================================================

        data["docs"] = final_docs

        return super().handle(data)