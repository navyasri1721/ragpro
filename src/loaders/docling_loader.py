from docling.document_converter import DocumentConverter

class DoclingLoader:

    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, file_path):

        result = self.converter.convert(file_path)

        doc = result.document

        text = ""
        for block in doc.blocks:
            if hasattr(block, "text"):
                text += block.text + "\n"

        for table in doc.tables:
            text += str(table) + "\n"

        return text