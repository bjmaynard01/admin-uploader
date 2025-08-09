from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import uuid
import psycopg2
import requests
import fitz  # PyMuPDF for PDF parsing

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'md'}

# Configure your DB connection and embedding endpoint here
PG_CONN = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST'),
    'port': int(os.getenv('PG_PORT', 5432))
}

EMBEDDING_ENDPOINT = 'http://ollama:11434/api/embeddings'
GLOBAL_COLLECTION_NAME = 'global_docs'

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_collection_id():
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (GLOBAL_COLLECTION_NAME,))
    row = cur.fetchone()
    if not row:
        new_id = str(uuid.uuid4())
        cur.execute("INSERT INTO langchain_pg_collection (uuid, name, cmetadata) VALUES (%s, %s, %s)",
                    (new_id, GLOBAL_COLLECTION_NAME, '{}'))
        conn.commit()
        collection_id = new_id
    else:
        collection_id = row[0]
    cur.close()
    conn.close()
    return collection_id

def parse_pdf(filepath):
    doc = fitz.open(filepath)
    text_chunks = []
    for page in doc:
        text = page.get_text()
        if text:
            text_chunks.extend(text.split('\n\n'))  # crude chunking by paragraph
    return [chunk.strip() for chunk in text_chunks if chunk.strip()]

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

def embed_text(text):
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    try:
        response = requests.post(EMBEDDING_ENDPOINT, json=payload)
        response.raise_for_status()
        payload = response.json()
        if 'data' in payload:
            return payload['data'][0]['embedding']  # OpenAI-style fallback
        elif 'embedding' in payload:
            return payload['embedding']  # Ollama single-embedding response
        else:
            print(f"Unknown embedding response: {payload}")
            raise ValueError("Embedding response missing expected fields.")
    except Exception as e:
        print(f"Embedding error: {e}")
        print(f"Full response: {response.status_code} {response.text}")
        raise

def store_embedding(chunk, embedding, collection_id, custom_id_prefix):
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO langchain_pg_embedding (uuid, collection_id, document, embedding, cmetadata, custom_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), collection_id, chunk, embedding, 
          '{"admin_uploaded": true}', f'{custom_id_prefix}-{uuid.uuid4()}'))
    print(f"Storing chunk: {chunk[:60]}... with custom_id {custom_id_prefix}")

    conn.commit()
    cur.close()
    conn.close()

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
            chunks = parse_pdf(filepath)
            for i, chunk in enumerate(chunks):
                try:
                    emb = embed_text(chunk)
                    store_embedding(chunk, emb, collection_id, filename)
                except Exception as e:
                    print(f"[ERROR] Failed on chunk {i}: {e}")
                    continue
            flash(f"Uploaded and processed {filename} with {len(chunks)} chunks.")
            return redirect(url_for('upload_file'))
    return render_template('upload.html')
