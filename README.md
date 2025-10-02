# 📄✨ Local/Online Document Summarizer

Summarize long-form `.pdf` and `.docx` files with Hugging Face models from a single Docker image. Use local model weights or the Hugging Face Inference API, and optionally build temporary FAISS vector stores you can delete when you're done.

# NOTE
I am currently working on this project, experimenting developing habits and new stuff. Please consider this code as experimental.
I'll gladly inform when it's gonna be stable. **thanks for your time**.

## ✨ Features

- Docker-only workflow; no Python or system packages to install locally.
- Three summarization presets: small (`sshleifer/distilbart-cnn-12-6`), medium (`facebook/bart-large-cnn`), large (`google/pegasus-cnn_dailymail`).
- Supports `.pdf` and `.docx` documents.
- Optional FAISS vector store per document for quick re-querying.
- Hugging Face API support via `HF_API_TOKEN`.

## 🧰 Prerequisites

### 🪟 Windows

- Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
- Enable WSL 2 during setup and ensure virtualization is turned on in BIOS if prompted.
- Open PowerShell and run `docker version` to confirm Docker is running.
- (Optional) Install [Git for Windows](https://git-scm.com/download/win) for convenient cloning.

### 🍎 macOS

- Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/), choosing the Intel or Apple Silicon build that matches your hardware.
- Launch Docker Desktop once so the background service stays active (the icon should appear in the menu bar).
- Open Terminal and run `docker version` to confirm Docker is ready.
- (Optional) Install [Homebrew](https://brew.sh/) if you also want `git`.

### 🐧 Linux

- Install Docker Engine following the [official guides](https://docs.docker.com/engine/install/) for your distribution.
- Add your user to the `docker` group and re-login: `sudo usermod -aG docker $USER`.
- Make sure the daemon is running: `sudo systemctl enable --now docker` (adjust for non-systemd distros).
- Verify with `docker version`; install `git` via your package manager if needed.

## 🐳 Quick Start with Docker

1. Clone the repository and build the image:

   ```bash
   git clone https://github.com/<your-account>/local_llm.git
   cd local_llm
   docker build -t local-llm .
   ```

2. Summarize a document from your host machine (replace the path with your file):

   ```bash
   docker run --rm \
     -v "$(pwd)":/workspace \
     -w /workspace \
     local-llm path/to/file.pdf --mode local --size medium --out summary.txt
   ```

   The summary is written back into your project folder. Choose `--size small|medium|large` to switch models.

### 🔐 Use the Hugging Face API

Provide a token to call the hosted inference API instead of local weights:

```bash
docker run --rm \
  -e HF_API_TOKEN=hf_your_token_here \
  -v "$(pwd)":/workspace \
  -w /workspace \
  local-llm path/to/file.pdf --mode api --size large --out api_summary.txt
```

### 💾 Cache Models Between Runs

Persist downloaded model files so subsequent runs start instantly:

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -v local_llm_models:/app/models \
  local-llm path/to/file.pdf --mode local --size medium --download --out summary.txt
```

You can reuse the same named volume (`local_llm_models`) across runs. Skip `--download` once the cache exists.

### 🔍 Use the Vector Store

Build a temporary FAISS index for chunk-level lookups, and optionally delete it when finished:

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  local-llm path/to/file.pdf --mode local --size medium --use-vdb --out summary.txt
```

Add `--delete-vdb` to remove the index immediately after summarization, or delete the `vdb/` folder manually later.

### 📦 Pre-download Models Without Summarizing

```bash
docker run --rm \
  -v local_llm_models:/app/models \
  --entrypoint python \
  local-llm download_models.py --size large
```

## 🧠 How It Works

- `summarizer/documents.py` extracts text from `.pdf` (PyPDF) or `.docx` (python-docx).
- `summarizer/summarize.py` chunks the text, summarizes each chunk, and merges the results.
  - Local mode uses `transformers.pipeline("summarization")`.
  - API mode POSTs to `https://api-inference.huggingface.co/models/<model>`.
- Optional FAISS indices are stored in `vdb/<doc>` when `--use-vdb` is passed.

## ⚠️ Notes

- `.doc` files are not supported; convert to `.docx` first.
- Large documents benefit from enabling GPU passthrough if your Docker setup supports it.
- Delete the `vdb/` folder or Docker volumes when you no longer need cached data.
