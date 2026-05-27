from langchain_huggingface import HuggingFaceEmbeddings

class CustomHFEmbeddings:

    def get(self):
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )