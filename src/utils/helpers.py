from langchain_core.documents import (
    Document
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# =========================================================
# SMART SPLITTER
# =========================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1200,

        chunk_overlap=200,

        separators=[

            "\n\n",

            "\n",

            ". ",

            " ",

            ""
        ]
    )

    split_docs = []

    for doc in documents:

        try:

            text = doc.page_content

            metadata = doc.metadata

            # =====================================
            # TABLE DOCUMENTS
            # =====================================

            if metadata.get("type") == "table":

                rows = text.split("\n")

                for row in rows:

                    row = row.strip()

                    if len(row) < 5:
                        continue

                    # =================================
                    # NORMALIZE TABLE TERMS
                    # =================================

                    normalized_row = normalize_table_row(
                        row
                    )

                    split_docs.append(

                        Document(

                            page_content=normalized_row,

                            metadata=metadata
                        )
                    )

            # =====================================
            # NORMAL TEXT DOCUMENTS
            # =====================================

            else:

                chunks = splitter.split_text(
                    text
                )

                for chunk in chunks:

                    chunk = chunk.strip()

                    if len(chunk) < 20:
                        continue

                    split_docs.append(

                        Document(

                            page_content=chunk,

                            metadata=metadata
                        )
                    )

        except Exception as e:

            print(
                f"[DEBUG] SPLIT ERROR: {e}"
            )

    print(
        f"[DEBUG] SPLIT DOCS: {len(split_docs)}"
    )

    return split_docs


# =========================================================
# TABLE NORMALIZATION
# =========================================================

def normalize_table_row(row):

    row_lower = row.lower()

    enriched = row

    # =====================================================
    # BOND NORMALIZATION
    # =====================================================

    if "bond" in row_lower:

        if "| 0 |" in row_lower or "0 years" in row_lower:

            enriched += (
                " ; bond-free company ; no bond"
            )

    # =====================================================
    # BACKLOG NORMALIZATION
    # =====================================================

    if "backlog" in row_lower:

        if "| 0 |" in row_lower:

            enriched += (
                " ; no backlogs allowed"
            )

        if "| 1 |" in row_lower:

            enriched += (
                " ; allows 1 backlog"
            )

        if "| 2 |" in row_lower:

            enriched += (
                " ; allows 2 backlogs"
            )

    # =====================================================
    # PACKAGE NORMALIZATION
    # =====================================================

    if "lpa" in row_lower or "package" in row_lower:

        enriched += (
            " ; salary package information"
        )

    # =====================================================
    # CGPA NORMALIZATION
    # =====================================================

    if "cgpa" in row_lower:

        enriched += (
            " ; eligibility criteria"
        )

    # =====================================================
    # INTERVIEW NORMALIZATION
    # =====================================================

    if "interview" in row_lower:

        enriched += (
            " ; interview preparation"
        )

    return enriched