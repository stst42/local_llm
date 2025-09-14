import argparse
import hashlib
import os
import shutil
from pathlib import Path
import requests
import re

from summarizer.documents import load_text_from_file
from summarizer.models import build_local_summarizer, download_model, get_model_id
from summarizer.vector_store import build_vector_store


def clean_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        if len(ln.split()) < 3 and not re.search(r"[\.!?]$", ln):
            continue
        kept.append(ln)
    return " ".join(kept)


def split_into_chunks(text: str, max_chars: int = 2500):
    text = clean_text(text)
    sents = re.split(r"(?<=[\.!?])\s+", text)
    chunks = []
    cur = ""
    for s in sents:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def summarize_with_api(
    model_id: str,
    text: str,
    token: str,
    max_length: int = 200,
    min_length: int = 60,
    num_beams: int = 4,
):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "do_sample": False,
            "no_repeat_ngram_size": 3,
            "length_penalty": 2.0,
        },
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
    p.add_argument("path", help="Path to .pdf/.docx or .txt file")
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
        "--gpu",
        action="store_true",
        help="Use GPU if available (local mode)",
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
            summarizer = build_local_summarizer(local_dir, use_gpu=args.gpu)
        else:
            summarizer = build_local_summarizer(model_id, use_gpu=args.gpu)

        def summarize_local_chunk(ch: str) -> str:
            w = len(ch.split())
            if w < 150:
                mx, mn = 120, 30
            elif w < 400:
                mx, mn = 180, 60
            else:
                mx, mn = 220, 80
            out = summarizer(
                ch,
                max_length=mx,
                min_length=mn,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
            )[0]["summary_text"]
            return out

        interim = [summarize_local_chunk(ch) for ch in chunks]
        combined = "\n".join(interim)
        final = summarizer(
            combined,
            max_length=220,
            min_length=80,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )[0]["summary_text"]
    else:
        token = os.environ.get("HF_API_TOKEN", "")
        interim = []
        for ch in chunks:
            w = len(ch.split())
            if w < 150:
                mx, mn = 120, 30
            elif w < 400:
                mx, mn = 180, 60
            else:
                mx, mn = 220, 80
            s = summarize_with_api(model_id, ch, token, max_length=mx, min_length=mn)
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
