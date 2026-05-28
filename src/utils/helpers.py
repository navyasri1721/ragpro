from langchain_core.documents import (
    Document
)

# =====================================================
# MANUAL DOCUMENT SPLITTER
# =====================================================

def split_documents(

    documents,

    chunk_size=1000,

    chunk_overlap=200
):

    split_docs = []

    for doc in documents:

        text = doc.page_content

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunk_doc = Document(

                page_content=chunk_text,

                metadata=doc.metadata
            )

            split_docs.append(
                chunk_doc
            )

            start += (
                chunk_size - chunk_overlap
            )

    return split_docs