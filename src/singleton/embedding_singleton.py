from src.embeddings.hf_embeddings import CustomHFEmbeddings

class EmbeddingSingleton:

    _instance = None

    @staticmethod
    def get_instance():

        if EmbeddingSingleton._instance is None:
            EmbeddingSingleton._instance = CustomHFEmbeddings().get()

        return EmbeddingSingleton._instance