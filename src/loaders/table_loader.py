import pdfplumber
from langchain_core.documents import Document
from io import BytesIO


def extract_tables_from_pdf(file_bytes, filename="file.pdf"):
    docs = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:

        for page_num, page in enumerate(pdf.pages):

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                table_text = ""

                for row in table:
                    row = [str(cell) if cell else "" for cell in row]
                    table_text += " | ".join(row) + "\n"

                docs.append(Document(
                    page_content=table_text,
                    metadata={
                        "source": filename,
                        "type": "table",
                        "page": page_num + 1
                    }
                ))

    return docs