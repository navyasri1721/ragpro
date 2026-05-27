def rewrite_query(
    question,
    memory=None
):

    if memory:

        history = memory.load_memory_variables(
            {}
        )

    rewritten_question = (
        question.strip()
    )

    return rewritten_question