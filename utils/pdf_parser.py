from pypdf import PdfReader
from typing import BinaryIO

def extract_text_from_pdf_stream(stream: BinaryIO) -> str:
    reader = PdfReader(stream)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
