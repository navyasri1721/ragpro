def refine_context(docs):

    refined_chunks = []

    seen_chunks = set()

    for doc in docs:

        content = (
            doc.page_content.strip()
        )

        if content not in seen_chunks:

            refined_chunks.append(
                content
            )

            seen_chunks.add(
                content
            )

    refined_context = "\n\n".join(
        refined_chunks
    )

    return refined_context