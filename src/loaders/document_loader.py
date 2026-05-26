import os
from src.loaders.docling_loader import DoclingLoader

class TXTLoader:
    def load(self, path):
        return open(path, "r", encoding="utf-8").read()

class PDFLoader:
    def load(self, path):
        return DoclingLoader().load(path)

class DOCXLoader:
    def load(self, path):
        return DoclingLoader().load(path)

class CSVLoader:
    def load(self, path):
        import pandas as pd
        return pd.read_csv(path).to_string()


def get_loader(ext):

    return {
        ".pdf": PDFLoader(),
        ".txt": TXTLoader(),
        ".docx": DOCXLoader(),
        ".csv": CSVLoader()
    }.get(ext)


def process_uploaded_files(files):

    docs = []
    os.makedirs("data/uploads", exist_ok=True)

    for f in files:

        path = f"data/uploads/{f.name}"
        with open(path, "wb") as file:
            file.write(f.getbuffer())

        ext = os.path.splitext(path)[1]

        loader = get_loader(ext)

        if loader:
            docs.append({
                "content": loader.load(path),
                "source": f.name
            })

    return docs