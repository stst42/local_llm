# Local/Online Document Summarizer

An experiment for a simple, no-frills project to summarize `.pdf` or `.docx` files using Hugging Face models. Run the model locally (download weights) or via the Hugging Face Inference API. Optionally builds a lightweight local vector database of document chunks that you can delete after use. This project is on the way to be completed.

## Features

- Local or API inference
- Three model sizes:
  - small: `sshleifer/distilbart-cnn-12-6`
  - medium: `facebook/bart-large-cnn`
  - large: `google/pegasus-cnn_dailymail`
- Supports `.pdf` and `.docx`
- Optional local vector store with FAISS (one dir per processed doc)

## Quick Start (Local Python)

1) Install Python dependencies

```bash
pip install -r requirements.txt
```

2) (Optional) Use HF API mode by setting a token

```bash
export HF_API_TOKEN=hf_your_token_here
```

3) (Optional) Pre-download local models

```bash
python download_models.py --size small
python download_models.py --size medium
python download_models.py --size large
```

4) Run summarization

- Local inference (downloads on first use if not pre-downloaded):

```bash
python -m summarizer.summarize path/to/file.pdf --mode local --size medium --out summary.txt
```

- Local inference with forced pre-download during run:

```bash
python -m summarizer.summarize path/to/file.docx --mode local --size small --download --out my_summary.txt
```

- HF Inference API mode:

```bash
python -m summarizer.summarize path/to/file.pdf --mode api --size large --out api_summary.txt
```

5) Vector database (optional)

- Create a local FAISS vector store for the document chunks:

```bash
python -m summarizer.summarize path/to/file.pdf --use-vdb --out summary.txt
```

- Delete the vector store after summarization:

```bash
python -m summarizer.summarize path/to/file.pdf --use-vdb --delete-vdb --out summary.txt
```

## Docker Workflow

### 1. Clone the repository and build the image

```bash
git clone https://github.com/<your-account>/local_llm.git
cd local_llm
docker build -t local-llm .
```

### 2. Inspect the CLI inside the container

```bash
docker run --rm local-llm --help
```

### 3. Summarize documents from your host machine

- Mount the folder that contains the documents you want to process (here we mount the repository root and set it as the working directory).
- Mount a named volume at `/app/models` so model downloads are cached across runs when you pass `--download`.

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -v local_llm_models:/app/models \
  local-llm path/to/file.pdf --mode local --size medium --download --out summary.txt
```

On the first run the container downloads the model weights into the mounted volume. Subsequent runs can omit `--download` and reuse the cached model copy.

### 4. (Optional) Persist the Hugging Face cache instead of using `--download`

If you prefer the default Hugging Face cache, mount a volume at `/root/.cache/huggingface` and skip `--download`:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -v local_llm_cache:/root/.cache/huggingface \
  local-llm path/to/file.pdf --mode local --size medium --out summary.txt
```

### 5. (Optional) Provide a Hugging Face token for API mode

```bash
docker run --rm \
  -e HF_API_TOKEN=hf_your_token_here \
  -v $(pwd):/workspace \
  -w /workspace \
  local-llm path/to/file.pdf --mode api --size large --out api_summary.txt
```

### 6. (Optional) Pre-download models without running a summarization job

```bash
docker run --rm \
  -v local_llm_models:/app/models \
  --entrypoint python \
  local-llm download_models.py --size large
```

The summary file and any vector store output (`vdb/...`) are written into the mounted workspace directory so they persist on the host.

## How It Works (simple)

- `summarizer/documents.py` loads text from `.pdf` (PyPDF) and `.docx` (python-docx)
- `summarizer/summarize.py` splits text into overlapping chunks and summarizes each
  - Local mode: `transformers.pipeline("summarization")`
  - API mode: POST to `https://api-inference.huggingface.co/models/<model>`
- Summaries are combined and summarized again for a shorter final output
- If `--use-vdb` is passed, `summarizer/vector_store.py` builds a FAISS index in `vdb/<doc>`

## Notes

- `.doc` is not supported (no `textract`). Convert `.doc` to `.docx` if needed.
- For large documents, local inference benefits from a GPU and proper PyTorch install.
- The code is intentionally minimal and straightforward.
