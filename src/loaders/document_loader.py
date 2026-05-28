import os

from langchain_core.documents import (
    Document
)

# =====================================================
# TXT LOADER
# =====================================================

class TXTLoader:

    def load(self, path):

        try:

            with open(

                path,

                "r",

                encoding="utf-8",

                errors="ignore"

            ) as file:

                return file.read()

        except:

            return ""

# =====================================================
# PDF LOADER
# =====================================================

class PDFLoader:

    def load(self, path):

        try:

            import fitz

            doc = fitz.open(path)

            text = ""

            for page in doc:

                text += page.get_text()

            return text

        except:

            return ""

# =====================================================
# DOCX LOADER
# =====================================================

class DOCXLoader:

    def load(self, path):

        try:

            from docx import Document as DocxDocument

            doc = DocxDocument(path)

            text = "\n".join([

                para.text

                for para in doc.paragraphs
            ])

            return text

        except:

            return ""

# =====================================================
# CSV LOADER
# =====================================================

class CSVLoader:

    def load(self, path):

        try:

            import pandas as pd

            dataframe = pd.read_csv(path)

            return dataframe.to_string()

        except:

            return ""

# =====================================================
# GET LOADER
# =====================================================

def get_loader(ext):

    return {

        ".pdf": PDFLoader(),

        ".txt": TXTLoader(),

        ".docx": DOCXLoader(),

        ".csv": CSVLoader()

    }.get(ext)

# =====================================================
# PROCESS FILES
# =====================================================

def process_uploaded_files(files):

    documents = []

    os.makedirs(

        "data/uploads",

        exist_ok=True
    )

    for file in files:

        path = f"data/uploads/{file.name}"

        with open(path, "wb") as f:

            f.write(file.getbuffer())

        ext = os.path.splitext(path)[1].lower()

        loader = get_loader(ext)

        if loader:

            content = loader.load(path)

            # =====================================
            # SKIP EMPTY CONTENT
            # =====================================

            if (

                content

                and content.strip()
            ):

                document = Document(

                    page_content=content,

                    metadata={

                        "source": file.name
                    }
                )

                documents.append(
                    document
                )

    return documents