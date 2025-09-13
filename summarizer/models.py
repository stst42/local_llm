from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


ModelSize = Literal["small", "medium", "large"]


MODEL_MAP = {
    "small": "sshleifer/distilbart-cnn-12-6",
    "medium": "facebook/bart-large-cnn",
    "large": "google/pegasus-cnn_dailymail",
}


def get_model_id(size: ModelSize) -> str:
    return MODEL_MAP[size]


def download_model(size: ModelSize, target_dir: str = "models") -> str:
    model_id = get_model_id(size)
    out_dir = Path(target_dir) / model_id.replace("/", "--")
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id, local_dir=str(out_dir), local_dir_use_symlinks=False
    )
    # Also warm up transformers cache
    AutoTokenizer.from_pretrained(str(out_dir))
    AutoModelForSeq2SeqLM.from_pretrained(str(out_dir))
    return str(out_dir)


def build_local_summarizer(model_dir_or_id: str):
    tok = AutoTokenizer.from_pretrained(model_dir_or_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir_or_id)
    return pipeline("summarization", model=mdl, tokenizer=tok)
