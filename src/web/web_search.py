from duckduckgo_search import DDGS


def search_web(query):

    results = []

    try:

        with DDGS() as ddgs:

            response = ddgs.text(
                query,
                max_results=5
            )

            for r in response:

                results.append({

                    "title": r.get(
                        "title",
                        ""
                    ),

                    "body": r.get(
                        "body",
                        ""
                    ),

                    "href": r.get(
                        "href",
                        ""
                    )
                })

    except Exception as e:

        print(
            f"WEB SEARCH ERROR: {e}"
        )

    return results