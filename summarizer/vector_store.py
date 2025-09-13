import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def embed_chunks(chunks, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    emb = SentenceTransformer(model_name)
    vecs = emb.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return vecs


def build_vector_store(chunks, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vecs = embed_chunks(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype(np.float32))
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    np.save(os.path.join(out_dir, "embeddings.npy"), vecs)
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"text": ch}) + "\n")
    return out_dir
