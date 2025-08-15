# Admin Uploader

The **Admin Uploader** is a private tool for ingesting VA-related source documents into a Postgres + PGVector database for use with Retrieval-Augmented Generation (RAG) chatbots.

It provides:
- A Flask web UI for uploading and processing files.
- CLI scripts for batch processing and reindexing.
- Consistent chunking, embedding, and metadata for accurate chatbot citations.
- Extensible architecture for adding new ingestion sources.

> **Important:** This is an **admin-only** service. Do not expose it to the public internet.

---

## 📂 Repository Structure

```
admin_uploader/
├── app.py                  # Flask app entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment configuration
├── adapters/               # (Optional) Source-specific ingestion adapters
├── core/                   # Shared chunking, normalization, and DB logic
├── static/                 # Flask static assets
├── templates/              # Flask HTML templates
├── uploads/                # Uploaded files (gitignored)
├── reindex.py              # Rebuild vector index from JSONL chunks
├── manage_uploads.py       # CLI for batch ingestion
└── README.md               # This file
```

---

## ⚙️ Setup

### 1. Clone and Install
```bash
git clone https://github.com/yourname/admin_uploader.git
cd admin_uploader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and edit values:
```env
FLASK_ENV=development
SECRET_KEY=change-me

# Upload handling
UPLOAD_FOLDER=uploads
ALLOWED_EXT=pdf,doc,docx,txt,md
MAX_CONTENT_LENGTH_MB=50

# Database
PG_DBNAME=postgres
PG_USER=postgres
PG_PASSWORD=postgres
PG_HOST=localhost
PG_PORT=5432

# Vector collection
VECTOR_COLLECTION=global_docs

# Embedding model (must match your chatbot’s query-time model)
EMBED_MODEL=nomic-embed-text-v1.5
```

### 3. Enable pgvector in Postgres
From `psql`:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 🚀 Usage

### Web UI
Run:
```bash
flask --app app run --port 8080
```
Then open: http://127.0.0.1:8080

Workflow:
1. Upload a file (PDF, DOCX, TXT, or MD).
2. The app parses, cleans, and chunks content.
3. Each chunk is embedded and upserted into the PGVector collection.
4. Optional HTML snapshots can be reviewed for QA.

### CLI: Batch Uploads
```bash
python manage_uploads.py   --path ./incoming   --collection global_docs   --model nomic-embed-text-v1.5
```
Options:
- `--dry-run` → process but don’t write to DB
- `--batch N` → set upsert batch size
- `--snapshots` → save HTML snapshot for each processed file

### CLI: Reindex
```bash
python reindex.py   --input out/m21_1.jsonl   --collection global_docs   --model nomic-embed-text-v1.5
```
Flags:
- `--truncate` → clear collection before inserting
- `--dry-run` → print stats without DB writes
- `--batch N` → batch size for DB upserts

---

## 🧩 How It Works

1. **Upload** → File saved in `UPLOAD_FOLDER` with UUID session.
2. **Parse** → PDF/DOCX/TXT/MD read and cleaned, headings/lists preserved.
3. **Chunk** → Heading-aware, ~1–2.5k tokens with overlap.
4. **Embed** → Model specified in `EMBED_MODEL` env var.
5. **Upsert** → Stable `id` from `(url_or_filename, text_hash)`.
6. **QA** → Optional HTML snapshots.

---

## 📦 Docker Compose Example
```yaml
version: "3.9"
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
  uploader:
    build: .
    env_file: .env
    ports:
      - "8080:8080"
    volumes:
      - ./uploads:/app/uploads
volumes:
  pgdata:
```

---

## 🔐 Security
- Restrict network access to trusted admins.
- Use HTTPS in production.
- Store credentials in `.env` (never commit).
- Consider reverse proxy with SSO or basic auth.

---

## 🛠 Extending
- Add new ingestion logic in `adapters/`.
- Reuse `core/normalize.py` and `core/chunk.py`.
- Keep metadata consistent for smooth RAG retrieval.
