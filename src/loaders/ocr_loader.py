import pytesseract
from pdf2image import convert_from_bytes
from langchain_core.documents import Document

# set path ONCE
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_scanned_pdf(file_bytes, filename="file.pdf"):
    images = convert_from_bytes(file_bytes)

    text = ""

    for i, img in enumerate(images):
        text += f"\n--- Page {i+1} ---\n"
        text += pytesseract.image_to_string(img)

    return Document(
        page_content=text,
        metadata={"source": filename, "type": "ocr_pdf"}
    )


def extract_text_from_image(image, filename="image"):
    text = pytesseract.image_to_string(image)

    return Document(
        page_content="IMAGE OCR:\n" + text,
        metadata={"source": filename, "type": "image_ocr"}
    )