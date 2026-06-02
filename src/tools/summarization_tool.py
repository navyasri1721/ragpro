from langchain_groq import ChatGroq

import streamlit as st

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(

    groq_api_key=st.secrets[
        "GROQ_API_KEY"
    ],

    model_name="llama-3.3-70b-versatile"
)

# =========================================================
# SUMMARIZATION TOOL
# =========================================================

def summarization_tool(context):

    # =========================================
    # EMPTY CHECK
    # =========================================

    if not context.strip():

        return "No content available."

    # =========================================
    # LIMIT HUGE CONTEXT
    # =========================================

    context = context[:12000]

    # =========================================
    # PROMPT
    # =========================================

    summary_prompt = f"""
You are a RAG-grade document intelligence system.

Your task is NOT to copy the document.

Your task is to ANALYZE it like a data scientist.

================================================
STRICT RULES:
================================================
- Do NOT repeat long raw tables
- Do NOT copy text directly
- Extract meaning, patterns, and insights
- If numbers conflict, mention the conflict
- Group similar companies together
- Highlight trends and comparisons
- Convert tables into insights
- Be concise but intelligent

================================================
OUTPUT FORMAT:
================================================

1. DOCUMENT TYPE
- What kind of dataset is this?

2. STRUCTURE OVERVIEW
- What sections exist?

3. KEY DATA INSIGHTS
- Important patterns in numbers/tables
- Comparisons between companies
- Extremes (highest/lowest values)

4. REASONING CHALLENGES
- Conflicts in data
- Multi-hop reasoning cases
- Ambiguities or contradictions

5. REAL-WORLD USE CASE
- What can this dataset be used for?

6. FINAL SUMMARY
- 3–5 lines high-quality conclusion

================================================
DOCUMENT:
{combined_text}
"""
    # =========================================
    # LLM CALL
    # =========================================

    response = llm.invoke(prompt)

    answer = (

        response.content

        if hasattr(response, "content")

        else str(response)
    )

    return answer.strip()