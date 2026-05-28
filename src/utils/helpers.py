from langchain_core.documents import Document

def split_documents(docs):

    split_docs = []

    for doc in docs:

        text = doc.page_content

        lines = text.split("\n")

        for line in lines:

            line = line.strip()

            if len(line) < 5:
                continue

            # IMPORTANT: each row becomes its own chunk
            split_docs.append(
                Document(
                    page_content=line,
                    metadata=doc.metadata
                )
            )

    return split_docs