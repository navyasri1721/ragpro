from langchain_groq import ChatGroq

class LLMFactory:

    @staticmethod
    def create_llm(api_key):

        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192"
        )