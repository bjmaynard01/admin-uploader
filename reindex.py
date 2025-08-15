#!/usr/bin/env python3
import os, json, uuid, argparse, psycopg2
from main import (
    PG_CONN, GLOBAL_COLLECTION_NAME,
    parse_file, embed_text, get_collection_id, store_embedding
)

def get_conn():
    return psycopg2.connect(**PG_CONN)

def collection_id(cur):
    cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name=%s", (GLOBAL_COLLECTION_NAME,))
    row = cur.fetchone()
    return row[0] if row else None

def cmd_list_sources(args):
    with get_conn() as conn, conn.cursor() as cur:
        cid = collection_id(cur)
        if not cid:
            print(f"No collection named {GLOBAL_COLLECTION_NAME}")
            return
        cur.execute("""
            SELECT cmetadata->>'source' AS source, COUNT(*) 
            FROM langchain_pg_embedding 
            WHERE collection_id=%s 
            GROUP BY source 
            ORDER BY COUNT(*) DESC;
        """, (cid,))
        rows = cur.fetchall()
        for src, n in rows:
            print(f"{src or 'UNKNOWN'}\t{n}")

def cmd_stats(args):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT name, uuid FROM langchain_pg_collection ORDER BY name;")
        cols = cur.fetchall()
        for name, uid in cols:
            cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id=%s", (uid,))
            n = cur.fetchone()[0]
            print(f"{name}\t{uid}\t{n} rows")

def cmd_delete_source(args):
    source = args.source
    confirm = args.yes
    with get_conn() as conn, conn.cursor() as cur:
        cid = collection_id(cur)
        if not cid:
            print(f"No collection named {GLOBAL_COLLECTION_NAME}")
            return
        cur.execute("""
            SELECT COUNT(*) FROM langchain_pg_embedding
            WHERE collection_id=%s AND cmetadata->>'source'=%s
        """, (cid, source))
        n = cur.fetchone()[0]
        if n == 0:
            print(f"No rows found for source: {source}")
            return
        if not confirm:
            print(f"About to DELETE {n} rows for source='{source}' in collection '{GLOBAL_COLLECTION_NAME}'. Re-run with --yes to confirm.")
            return
        cur.execute("""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id=%s AND cmetadata->>'source'=%s
        """, (cid, source))
        print(f"Deleted {n} rows for {source}")

def cmd_reindex_file(args):
    path = args.file
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    filename = os.path.basename(path)
    # Optional: delete existing
    if args.replace:
        args2 = argparse.Namespace(source=filename, yes=True)
        cmd_delete_source(args2)

    # Ensure collection exists
    cid = get_collection_id()

    # Parse + embed + store
    chunks = parse_file(path, filename)
    ok, fail = 0, 0
    for i, item in enumerate(chunks):
        try:
            emb = embed_text(item["text"])
            meta = {
                "admin_uploaded": True,
                "source": item["filename"],
                "page": item["page"],
                "section": item["section"],
                "content_type": item["content_type"],
            }
            store_embedding(item["text"], emb, cid, filename, meta)
            ok += 1
        except Exception as e:
            print(f"[ERROR] chunk {i}: {e}")
            fail += 1
    print(f"Reindex complete: {ok} ok, {fail} failed for {filename}")

def main():
    p = argparse.ArgumentParser(description="PGVector reindex utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-sources", help="List all sources and counts").set_defaults(func=cmd_list_sources)
    sub.add_parser("stats", help="Show collection stats").set_defaults(func=cmd_stats)

    d = sub.add_parser("delete-source", help="Delete all chunks for a source filename")
    d.add_argument("source", help="filename stored in cmetadata->>'source'")
    d.add_argument("--yes", action="store_true", help="confirm deletion")
    d.set_defaults(func=cmd_delete_source)

    r = sub.add_parser("reindex-file", help="Parse/embed/store a file")
    r.add_argument("file", help="path to file (pdf/txt/md/docx)")
    r.add_argument("--replace", action="store_true", help="delete existing rows for this filename first")
    r.set_defaults(func=cmd_reindex_file)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
