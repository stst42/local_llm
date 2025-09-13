import argparse
import hashlib
import os
import shutil
from pathlib import Path
import requests

from summarizer.documents import load_text_from_file
from summarizer.models import build_local_summarizer, download_model, get_model_id
from summarizer.vector_store import build_vector_store


def split_into_chunks(text: str, max_chars: int = 4000, overlap: int = 200):
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        chunk = text[i:end]
        chunks.append(chunk)
        if end == len(text):
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks


def summarize_with_api(
    model_id: str, text: str, token: str, max_length: int = 200, min_length: int = 60
):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_length, "min_length": min_length},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    data = r.json()
    if isinstance(data, list) and len(data) > 0 and "summary_text" in data[0]:
        return data[0]["summary_text"]
    if isinstance(data, dict) and "error" in data:
        return data["error"]
    return str(data)


def main():
    p = argparse.ArgumentParser(description="Summarize documents locally or via HF API")
    p.add_argument("path", help="Path to .pdf/.doc/.docx or .txt file")
    p.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Model size to use",
    )
    p.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help="Run locally or via HF Inference API",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download chosen model ahead of running (local mode)",
    )
    p.add_argument(
        "--use-vdb",
        action="store_true",
        help="Build a simple local vector DB of chunks",
    )
    p.add_argument(
        "--delete-vdb", action="store_true", help="Delete vector DB after summarization"
    )
    p.add_argument("--out", default="summary.txt", help="Where to save the summary")
    args = p.parse_args()

    text = load_text_from_file(args.path)
    chunks = split_into_chunks(text)

    base_name = Path(args.path).name
    doc_hash = hashlib.sha1(base_name.encode("utf-8")).hexdigest()[:8]
    vdb_dir = os.path.join("vdb", f"{base_name}.{doc_hash}")

    if args.use_vdb:
        build_vector_store(chunks, vdb_dir)

    model_id = get_model_id(args.size)

    if args.mode == "local":
        if args.download:
            local_dir = download_model(args.size)
            summarizer = build_local_summarizer(local_dir)
        else:
            summarizer = build_local_summarizer(model_id)

        interim = []
        for ch in chunks:
            s = summarizer(ch, max_length=200, min_length=60)[0]["summary_text"]
            interim.append(s)
        combined = "\n".join(interim)
        final = summarizer(combined, max_length=220, min_length=80)[0]["summary_text"]
    else:
        token = os.environ.get("HF_API_TOKEN", "")
        interim = []
        for ch in chunks:
            s = summarize_with_api(model_id, ch, token)
            interim.append(s)
        combined = "\n".join(interim)
        final = summarize_with_api(
            model_id, combined, token, max_length=220, min_length=80
        )

    Path(args.out).write_text(final, encoding="utf-8")

    if args.delete_vdb and os.path.isdir(vdb_dir):
        shutil.rmtree(vdb_dir)

    print("Summary saved to:", args.out)


if __name__ == "__main__":
    main()
