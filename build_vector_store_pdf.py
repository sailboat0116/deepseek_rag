import os
import faiss
import pickle
import PyPDF2
from sentence_transformers import SentenceTransformer

def load_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def chunk_text(text, max_length=500):
    sentences = text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_length:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

if __name__ == "__main__":
    file_path = "lung-rads-assessment-categories.pdf"
    text = load_text_from_pdf(file_path)
    chunks = chunk_text(text)
    print(f"📄 共有 {len(chunks)} 段文字")

    index, docs = build_faiss_index(chunks)
    faiss.write_index(index, "vector.index")
    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("✅ 向量資料庫已建立")
