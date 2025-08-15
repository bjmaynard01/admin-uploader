from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import re
import uuid
import json
import psycopg2
import requests
import fitz  # PyMuPDF for PDF parsing
import pdfplumber  # For table extraction
from flask import jsonify

try:
    import docx  # python-docx (optional for .docx)
except ImportError:
    docx = None

# Optional OCR
ENABLE_OCR = os.getenv("ENABLE_OCR", "0") == "1"
if ENABLE_OCR:
    import pytesseract
    from PIL import Image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'md'}

# ------------------------------
# CONFIG (env-driven)
# ------------------------------
PG_CONN = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST'),
    'port': int(os.getenv('PG_PORT', 5432)),
}
EMBEDDING_ENDPOINT = os.getenv('EMBEDDING_ENDPOINT', 'http://ollama:11434/api/embeddings')
GLOBAL_COLLECTION_NAME = os.getenv('GLOBAL_COLLECTION_NAME', 'global_docs')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")  # matches query side

# Chunking knobs
CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", 900))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 180))
MIN_CHARS_PER_CHUNK = int(os.getenv("RAG_MIN_CHARS", 180))

# OCR knobs
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", 200))  # if page text < this, try OCR
OCR_DPI = int(os.getenv("OCR_DPI", 300))

# crude token estimate (≈4 chars/token)
def est_tokens(s: str) -> int:
    return max(1, len(s) // 4)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------
# PGVector helpers
# ------------------------------
def get_collection_id():
    conn = psycopg2.connect(**PG_CONN)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (GLOBAL_COLLECTION_NAME,))
            row = cur.fetchone()
            if not row:
                new_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO langchain_pg_collection (uuid, name, cmetadata) VALUES (%s, %s, %s)",
                    (new_id, GLOBAL_COLLECTION_NAME, json.dumps({}))
                )
                return new_id
            return row[0]
    finally:
        conn.close()

def store_embedding(chunk_text, embedding, collection_id, custom_id_prefix, meta):
    conn = psycopg2.connect(**PG_CONN)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO langchain_pg_embedding (uuid, collection_id, document, embedding, cmetadata, custom_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                collection_id,
                chunk_text,
                embedding,
                json.dumps(meta),
                f'{custom_id_prefix}-{uuid.uuid4()}'
            ))
    finally:
        conn.close()

# ------------------------------
# Embedding
# ------------------------------
def embed_text(text):
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    resp = requests.post(EMBEDDING_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embedding":[...]} ; some providers return {"data":[{"embedding":[...]}]}
    if 'embedding' in data:
        return data['embedding']
    if 'data' in data and data['data'] and 'embedding' in data['data'][0]:
        return data['data'][0]['embedding']
    raise ValueError(f"Unknown embedding response format: {data}")

# ------------------------------
# Parsing / Chunking (Gov-doc tuned)
# ------------------------------
HEADING_RE = re.compile(r"^\s*((?:[A-Z][A-Z \-\(\)\/]{2,})|(?:\d+(?:\.\d+)*\s+.+)|(?:[A-Z][a-z].{0,80}:))\s*$")

def split_by_headings(lines):
    """Group lines into sections by heading-like lines."""
    sections = []
    current = {"heading": None, "lines": []}
    for line in lines:
        if HEADING_RE.match(line.strip()):
            if current["lines"]:
                sections.append(current)
            current = {"heading": line.strip(), "lines": []}
        else:
            current["lines"].append(line)
    if current["heading"] or current["lines"]:
        sections.append(current)
    return sections

def consolidate_paragraphs(text):
    """Simple paragraph normalization: collapse excessive newlines, keep bullets."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_section_text(section_text, heading, chunk_tokens=CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP):
    """Token-size based chunking with overlap; keeps heading at top of each chunk."""
    if not section_text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", section_text) if p.strip()]
    chunks = []
    buffer = heading + "\n" if heading else ""

    def flush(buf):
        buf = consolidate_paragraphs(buf)
        return buf if len(buf) >= MIN_CHARS_PER_CHUNK else None

    for para in paragraphs:
        candidate = (buffer + ("\n\n" if buffer else "") + para).strip()
        if est_tokens(candidate) <= chunk_tokens:
            buffer = candidate
            continue

        # paragraph would overflow: flush buffer, then start new
        flushed = flush(buffer)
        if flushed:
            chunks.append(flushed)

        # if single paragraph too big, hard-wrap by sentences
        sentences = re.split(r"(?<=[\.\?\!])\s+", para)
        buf2 = heading + "\n" if heading else ""
        for sent in sentences:
            tmp = (buf2 + (" " if buf2 else "") + sent).strip()
            if est_tokens(tmp) <= chunk_tokens:
                buf2 = tmp
            else:
                if buf2:
                    chunks.append(consolidate_paragraphs(buf2))
                buf2 = (heading + "\n" + sent) if heading else sent
        if buf2:
            chunks.append(consolidate_paragraphs(buf2))
        buffer = heading + "\n" if heading else ""

        # add overlap by borrowing tail of last chunk
        if chunks:
            tail = chunks[-1]
            take = max(0, len(tail) - (chunk_tokens - overlap_tokens) * 4)
            overlap = tail[take:]
            buffer = (heading + "\n" if heading else "") + overlap

    if buffer:
        flushed = flush(buffer)
        if flushed:
            chunks.append(flushed)

    return chunks

def ocr_page(fitz_page):
    """Rasterize page and OCR to text."""
    # DPI scaling: PyMuPDF uses transformation matrices (72 DPI base)
    zoom = OCR_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
    # Convert pixmap to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text

def extract_pdf_blocks_with_tables(filepath, filename):
    """
    Returns list of dicts: {text, page, section, content_type, filename}
    - text extraction via PyMuPDF
    - table extraction via pdfplumber (as TSV-ish lines)
    - OCR fallback if enabled & page text is sparse
    """
    results = []
    doc = fitz.open(filepath)

    # 1) text per page (with OCR fallback if needed)
    page_texts = []
    for pno, page in enumerate(doc, start=1):
        txt = page.get_text("text") or ""
        # Fallback to OCR if page looks like an image (low text signal)
        if ENABLE_OCR and len(txt.strip()) < OCR_MIN_TEXT_CHARS:
            try:
                ocr_txt = ocr_page(page)
                if len(ocr_txt.strip()) >= OCR_MIN_TEXT_CHARS:
                    txt = (txt + "\n\nOCR:\n" + ocr_txt).strip() if txt.strip() else ocr_txt
            except Exception as e:
                print(f"[WARN] OCR failed on page {pno}: {e}")
        # normalize bullets that PDFs sometimes split oddly
        txt = txt.replace("•\n", "• ").replace("\u2022\n", "\u2022 ")
        page_texts.append((pno, txt))

    # 2) tables per page (vector/extracted)
    table_text_by_page = {}
    with pdfplumber.open(filepath) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            lines = []
            for t in tables or []:
                for row in t:
                    row = [c.strip() if isinstance(c, str) else "" for c in row]
                    lines.append(" | ".join(row))
                if lines:
                    lines.append("")  # spacer between tables
            if lines:
                table_text_by_page[pno] = "TABLE:\n" + "\n".join(lines)

    # 3) build sections per page, then chunk
    for pno, txt in page_texts:
        lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
        sections = split_by_headings(lines)

        # If OCR added, optionally tag a lightweight OCR section (we keep it blended in the text)
        # Append table text at end of page as its own section (keeps retrieval simple)
        if pno in table_text_by_page:
            sections.append({"heading": "Tables", "lines": [table_text_by_page[pno]]})

        for sec in sections:
            heading = sec["heading"]
            sec_text = "\n".join(sec["lines"]).strip()
            if not sec_text:
                continue

            content_type = "table" if sec_text.startswith("TABLE:") or heading == "Tables" else "text"
            if "OCR:" in sec_text and content_type == "text":
                # soft marker that OCR contributed — handy for debugging/tuning
                content_type = "ocr"

            for chunk in chunk_section_text(sec_text, heading or ""):
                results.append({
                    "text": chunk,
                    "page": pno,
                    "section": heading or None,
                    "content_type": content_type,
                    "filename": filename,
                })

    return results

def parse_txt(filepath, filename):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
    chunks = []
    for blk in blocks:
        for ch in chunk_section_text(blk, heading=os.path.splitext(filename)[0]):
            chunks.append({
                "text": ch,
                "page": None,
                "section": None,
                "content_type": "text",
                "filename": filename,
            })
    return chunks

def parse_docx(filepath, filename):
    if docx is None:
        return parse_txt(filepath, filename)
    d = docx.Document(filepath)
    paras = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    text = "\n\n".join(paras)
    out = []
    for ch in chunk_section_text(text, heading=os.path.splitext(filename)[0]):
        out.append({
            "text": ch,
            "page": None,
            "section": None,
            "content_type": "text",
            "filename": filename,
        })
    return out

def parse_file(filepath, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return extract_pdf_blocks_with_tables(filepath, filename)
    if ext in ('txt', 'md'):
        return parse_txt(filepath, filename)
    if ext in ('docx',):
        return parse_docx(filepath, filename)
    # .doc fallback (treat as txt)
    return parse_txt(filepath, filename)

# ------------------------------
# Flask routes
# ------------------------------
@app.route('/admin/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            collection_id = get_collection_id()
            chunks = parse_file(filepath, filename)

            success, fail = 0, 0
            for i, item in enumerate(chunks):
                text = item["text"]
                try:
                    emb = embed_text(text)
                    meta = {
                        "admin_uploaded": True,
                        "source": item["filename"],
                        "page": item["page"],
                        "section": item["section"],
                        "content_type": item["content_type"],
                    }
                    store_embedding(text, emb, collection_id, filename, meta)
                    success += 1
                except Exception as e:
                    print(f"[ERROR] Failed on chunk {i}: {e}")
                    fail += 1
                    continue
            flash(f"Processed {filename}: {success} chunks stored, {fail} failed.")
            return redirect(url_for('upload_file'))
    return render_template('upload.html')

def _get_conn():
    return psycopg2.connect(**PG_CONN)

@app.route('/admin/stats', methods=['GET'])
def admin_stats():
    """High-level counts per collection and for your GLOBAL_COLLECTION_NAME."""
    try:
        data = {"collections": [], "global_docs": {"uuid": None, "total_rows": 0, "sources": []}}

        with _get_conn() as conn, conn.cursor() as cur:
            # All collections + counts
            cur.execute("SELECT name, uuid FROM langchain_pg_collection ORDER BY name;")
            cols = cur.fetchall()
            for name, uid in cols:
                cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id=%s;", (uid,))
                n = cur.fetchone()[0]
                data["collections"].append({"name": name, "uuid": uid, "rows": n})

            # Global collection breakdown by source
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name=%s;", (GLOBAL_COLLECTION_NAME,))
            row = cur.fetchone()
            if row:
                gid = row[0]
                data["global_docs"]["uuid"] = gid
                cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id=%s;", (gid,))
                data["global_docs"]["total_rows"] = cur.fetchone()[0]

                cur.execute("""
                    SELECT cmetadata->>'source' AS source, COUNT(*) AS n
                    FROM langchain_pg_embedding
                    WHERE collection_id=%s
                    GROUP BY source
                    ORDER BY n DESC NULLS LAST;
                """, (gid,))
                data["global_docs"]["sources"] = [{"source": s or "UNKNOWN", "rows": n} for (s, n) in cur.fetchall()]

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/sources', methods=['GET'])
def admin_sources():
    """Source-level stats for the global_docs collection (rows per source).
    Query params:
      - source: filter by source (exact or fuzzy)
      - exact:  '1' for exact match, default fuzzy (ILIKE %...%)
      - limit:  int
      - offset: int
      - order:  'rows' (default, desc) or 'source' (asc)
    """
    try:
        source_q = request.args.get('source', type=str)
        exact = request.args.get('exact', default='0') == '1'
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', type=int)
        order = request.args.get('order', default='rows')

        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name=%s;", (GLOBAL_COLLECTION_NAME,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": f"Collection {GLOBAL_COLLECTION_NAME} not found"}), 404
            gid = row[0]

            # Base WHERE
            where = ["collection_id=%s"]
            params = [gid]

            # Optional filter
            if source_q:
                if exact:
                    where.append("(cmetadata->>'source') = %s")
                    params.append(source_q)
                else:
                    where.append("(cmetadata->>'source') ILIKE %s")
                    params.append(f"%{source_q}%")

            where_sql = " AND ".join(where)

            # Ordering
            if order == "source":
                order_sql = "ORDER BY source ASC NULLS LAST"
            else:
                order_sql = "ORDER BY n DESC NULLS LAST"

            # Paging
            limit_sql = ""
            if isinstance(limit, int) and limit > 0:
                limit_sql += " LIMIT %s"
                params.append(limit)
                if isinstance(offset, int) and offset >= 0:
                    limit_sql += " OFFSET %s"
                    params.append(offset)

            cur.execute(f"""
                SELECT cmetadata->>'source' AS source, COUNT(*) AS n
                FROM langchain_pg_embedding
                WHERE {where_sql}
                GROUP BY source
                {order_sql}
                {limit_sql};
            """, tuple(params))

            results = [{"source": s or "UNKNOWN", "rows": n} for (s, n) in cur.fetchall()]
            return jsonify({"collection": GLOBAL_COLLECTION_NAME, "uuid": gid, "sources": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
