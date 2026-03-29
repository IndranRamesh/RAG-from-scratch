import os 
import sys 
from pypdf import PdfReader 
from typing import Any

# def load_pdf(path:str) -> list[dict]:
#     """Load a PDF and return a list of documents (one per page)."""

#     # fit the pdfreader function in this variable
#     reader = PdfReader(stream=path) 
#     documents = []

#     for i , page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text.strip(): # Strip empty pages
#             documents.append({
#                 "title":f"Page_{i+1}",
#                 "text":text
#             })
    
#     print(f"Loaded {len(documents)} pages from PDF")
#     return documents 

# def load_pdf(path: str) -> list:
#     reader = PdfReader(stream=path)
#     documents = []

#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if not text.strip():
#             continue

#         # Split by single newline, filter short lines
#         lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]

#         for j, line in enumerate(lines):
#             documents.append({
#                 "title": f"Page_{i+1}_Line_{j+1}",
#                 "text": line
#             })

#     print(f"Loaded {len(documents)} chunks from PDF")
#     return documents


def load_pdf(path: str) -> list:
    reader = PdfReader(stream=path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text.strip():
            continue

        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
        
        # Group every 5 lines into one chunk
        chunk_size = 5
        for j in range(0, len(lines), chunk_size):
            chunk = " ".join(lines[j:j+chunk_size])
            documents.append({
                "title": f"Page_{i+1}_Chunk_{j//chunk_size+1}",
                "text": chunk
            })

    print(f"Loaded {len(documents)} chunks from PDF")
    return documents

# To view the output in CMD
# -------------------------

# if __name__ == "__main__":
#     docs:list[dict[Any, Any]] = load_pdf(path=r"D:\bm25-war\Arul_ML_Metrics_And_Fundamentals_Guide.md.pdf")
#     for doc in docs[:1]:
#         print(f"\n{doc['title']}:")
#         print(doc['text'][:500])

if __name__ == "__main__":
    reader = PdfReader("Arul_ML_Metrics_And_Fundamentals_Guide.md.pdf")
    page = reader.pages[0]
    text = page.extract_text()
    print(repr(text[:500]))  # Shows actual newline characters