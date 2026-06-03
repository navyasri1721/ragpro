from collections import defaultdict

def group_by_source(docs):
    grouped = defaultdict(list)

    for d in docs:
        src = d.metadata.get("source", "unknown")
        grouped[src].append(d)

    return grouped


def pick_best_source(grouped, question):
    best_score = -1
    best_docs = []

    for src, docs in grouped.items():
        text = " ".join([d.page_content for d in docs])

        score = sum(
            1 for w in question.lower().split()
            if w in text.lower()
        )

        if score > best_score:
            best_score = score
            best_docs = docs

    return best_docs