import os

from langchain_core.documents import Document

import pytesseract

from src.loaders.table_loader import (
    extract_tables_from_pdf
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

            ) as f:

                return f.read()

        except:

            return ""


# =====================================================
# DOCX LOADER
# =====================================================

class DOCXLoader:

    def load(self, path):

        try:

            from docx import (
                Document as DocxDocument
            )

            doc = DocxDocument(path)

            return "\n".join([

                p.text

                for p in doc.paragraphs
            ])

        except:

            return ""


# =====================================================
# CSV LOADER
# =====================================================

class CSVLoader:

    def load(self, path):

        try:

            import pandas as pd

            df = pd.read_csv(path)

            return df.to_string(index=False)

        except:

            return ""


# =====================================================
# PDF LOADER
# =====================================================

class PDFLoader:

    def load(self, path):

        text = ""

        try:

            import fitz

            from pdf2image import (
                convert_from_path
            )

            doc = fitz.open(path)

            for page_num, page in enumerate(doc):

                page_text = page.get_text()

                # =====================================
                # NORMAL TEXT
                # =====================================

                if page_text:

                    text += f"""

PAGE_{page_num + 1}:

{page_text}
"""

                # =====================================
                # OCR PER PAGE
                # =====================================

                if len(page_text.strip()) < 100:

                    try:

                        images = convert_from_path(

                            path,

                            first_page=page_num + 1,

                            last_page=page_num + 1
                        )

                        for img in images:

                            ocr_text = (
                                pytesseract.image_to_string(
                                    img
                                )
                            )

                            text += f"""

OCR_PAGE_{page_num + 1}:

{ocr_text}
"""

                    except Exception as e:

                        print(
                            f"OCR PAGE ERROR: {e}"
                        )

        except Exception as e:

            print(
                f"PDF LOAD ERROR: {e}"
            )

            return ""

        return text


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

        try:

            file_bytes = file.getvalue()

            path = os.path.join(

                "data/uploads",

                file.name
            )

            with open(path, "wb") as f:

                f.write(file_bytes)

            ext = os.path.splitext(path)[1].lower()

            loader = get_loader(ext)

            if not loader:
                continue

            # =========================================
            # NORMAL DOCUMENT TEXT
            # =========================================

            content = loader.load(path)

            if content:

                content = str(content).strip()

                if len(content) > 20:

                    documents.append(

                        Document(

                            page_content=content,

                            metadata={

                                "source": file.name,

                                "type": ext.replace(".", ""),

                                "length": len(content)
                            }
                        )
                    )

            # =========================================
            # TABLE EXTRACTION
            # =========================================

            if ext == ".pdf":

                table_docs = (
                    extract_tables_from_pdf(

                        file_bytes,

                        file.name
                    )
                )

                documents.extend(
                    table_docs
                )

        except Exception as e:

            print(
                "FILE ERROR:",
                file.name,
                e
            )

    print(
        "[DEBUG] FINAL DOC COUNT:",
        len(documents)
    )

    return documents