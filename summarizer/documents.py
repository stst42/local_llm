import os
from pypdf import PdfReader
import docx


def load_text_from_pdf(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def load_text_from_docx(path):
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs]
    return "\n".join(paras)


def load_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_text_from_pdf(path)
    if ext == ".docx":
        return load_text_from_docx(path)
    return open(path, "r", encoding="utf-8").read()
