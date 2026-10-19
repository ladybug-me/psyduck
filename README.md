# Psyduck

Psyduck is an AI-powered fashion designer and outfit recommendation system. It combines multi-turn conversational intent extraction via local LLMs (Ollama) with vector-based semantic retrieval using FAISS and Sentence Transformers.

---

## Features

- Smart outfit suggestions: Multi-turn dialogue clarifies occasion, category, style, formality, color, and gender.
- Fast semantic search: FAISS vector indexing over thousands of curated fashion items.
- Local AI inference: Powered by Ollama (supports `llama3`, `qwen2.5-coder:3b`, `gemma3:1b`, `gpt-oss:120b-cloud`, etc.).
- Interactive web UI: Real-time chat with visual gallery cards, welcome video, and debug inspector.
- Automated server management: Comprehensive `run.sh` script with interactive TUI and command-line options.

---

## Quick Start (Automated Server Manager)

The recommended way to run and manage Psyduck is using `run.sh`:

```bash
git clone https://github.com/ladybug-me/psyduck.git
cd psyduck
chmod +x run.sh
./run.sh
```

Running `./run.sh` in a terminal opens the interactive management interface:

```
  ============================================================================
    PSYDUCK - Fashion Outfit Assistant Server Manager
  ============================================================================

System Status Overview:
  - Virtualenv (.venv)    : [OK] (Python 3.14)
  - Package Manager       : [OK] uv (high performance)
  - Direct Dependencies   : [OK] All packages installed
  - Fashion Dataset       : [OK] data.csv (44,441 images in database/)
  - SQLite Database       : [OK] fashion.db (44,441 records)
  - FAISS Vector Index    : [OK] fashion.index (44,441 vectors)
  - Ollama Service        : [OK] Active (v0.40.3)
------------------------------------------------------------------------------
Main Menu:
  1) Start Psyduck Server (uvicorn app:app)
  2) Setup & Maintenance Menu (Submenu)
  3) Verify Deployment Readiness
  4) Configure Active Model / Environment (.env)
  5) Exit
```

Selecting option **2 (Setup & Maintenance Menu)** opens the dedicated setup submenu:

```
--- Setup & Configuration Submenu ---

  1) Full Automated Setup (Dependencies, Dataset, Database, Ollama)
  2) Install / Update Python Dependencies (unpinned, uv/pip)
  3) Download Fashion Dataset only (KaggleHub -> data/)
  4) Build Database & FAISS Index only (data.csv -> fashion.db & index)
  5) Setup Ollama (install daemon & pull models)
  6) Verify Deployment Readiness
  7) Return to Main Menu
```

---

## Command-Line Automation

`run.sh` supports CLI options for headless deployments, background automation, and scripting:

```bash
# Start the server (optionally specifying model)
./run.sh
./run.sh llama3
./run.sh --run qwen2.5-coder:3b

# Access setup submenu or run full setup non-interactively
./run.sh --setup
./run.sh --full

# Install or update dependencies without version locks (using uv or pip)
./run.sh --deps

# Download fashion dataset only and place files in data/
./run.sh --download-data
./run.sh --force-download

# Build or rebuild SQLite database and FAISS vector index
./run.sh --build-db
./run.sh --force-build-db

# Install Ollama service and pull a model
./run.sh --ollama llama3
./run.sh --ollama qwen2.5-coder:3b

# Verify all required files and deployment readiness
./run.sh --check
```

---

## Manual Step-by-Step Setup

If you prefer configuring the project manually without `run.sh`:

### 1. Create and Activate Virtual Environment

Using `uv` (recommended):
```bash
uv venv .venv
source .venv/bin/activate
```

Or using standard `python3`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Direct Dependencies

Direct dependencies are specified without version locks (declared in `pyproject.toml`):

```bash
# Using uv
uv pip install fastapi uvicorn jinja2 requests python-dotenv sentence-transformers faiss-cpu numpy pandas kagglehub

# Or using pip
pip install fastapi uvicorn jinja2 requests python-dotenv sentence-transformers faiss-cpu numpy pandas kagglehub
```

### 3. Download Fashion Dataset

```bash
python3 data/download_data.py
```
This downloads `zara2099/fashion-style-image-dataset` via KaggleHub and organizes:
- `data/data.csv` (product catalog)
- `data/database/` (product photo images)

### 4. Build Database and FAISS Index

```bash
python3 data/build_database.py
```
This generates:
- `data/fashion.db` (SQLite product database)
- `data/fashion.index` (vector index for semantic search)

### 5. Configure Ollama Service

Psyduck uses Ollama for local intent triage and entity extraction:
```bash
sudo bash scripts/ollama_setup.sh --models llama3
```

Configure `.env` with host and model settings:
```env
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3
```

> **WSL Users:** When running Ollama on Windows and Psyduck inside WSL, set WSL networking to "Mirrored" and enable Host address loopback, or set `OLLAMA_HOST=<windows-ip>` in `.env`.

### 6. Start the Server

```bash
# Run with default model
python3 src/app.py

# Or specify a model at runtime
python3 src/app.py --model gemma3:1b

# Or run with uvicorn directly
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
```
http://127.0.0.1:8000
```

---

## Directory Structure

```
psyduck/
├── run.sh                       # Server runner and setup manager (TUI + CLI)
├── pyproject.toml               # Project configuration and direct dependencies
├── README.md                    # Documentation
├── src/
│   ├── app.py                   # Unified FastAPI backend application
│   ├── static/
│   │   ├── style.css            # Stylesheet
│   │   ├── easter_egg.jpg       # Asset
│   │   └── psyduck_video.mp4    # Welcome video
│   └── templates/
│       └── index.html           # Frontend template
├── scripts/
│   └── ollama_setup.sh          # Ollama daemon installer and model manager
└── data/
    ├── download_data.py         # KaggleHub dataset downloader
    ├── build_database.py        # SQLite DB and FAISS index builder
    ├── data.csv                 # Raw fashion metadata
    ├── fashion.db               # SQLite product database
    ├── fashion.index            # FAISS vector similarity index
    └── database/                # Product images directory
```

---

## Direct Dependencies

- **fastapi** & **uvicorn**: Async web framework and ASGI application server.
- **jinja2**: HTML template rendering.
- **requests**: HTTP client for interacting with Ollama.
- **python-dotenv**: Environment variable management.
- **sentence-transformers**: Sentence embeddings generation.
- **faiss-cpu**: Fast vector similarity search.
- **numpy** & **pandas**: Data manipulation and numerical operations.
- **kagglehub**: Dataset downloading.
