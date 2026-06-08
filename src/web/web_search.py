from tavily import TavilyClient
import streamlit as st
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
def search_web(query):
    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        results = []

        for r in response.get("results", []):

            results.append({
                "title": r.get("title", ""),
                "body": r.get("content", ""),
                "url": r.get("url", "")
            })

        return results

    except Exception as e:
        print("Tavily error occured:", e)
        return []