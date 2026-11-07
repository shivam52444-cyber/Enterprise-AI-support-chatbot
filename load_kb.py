"""
Load the KartNest policy files into Postgres (table kb_chunks) with embeddings.

Setup (once):
    pip install sentence-transformers "psycopg[binary]"

Run:
    python load_kb.py "D:\\path\\to\\folder\\with\\the\\md\\files"

What it does:
  1. Splits every .md file into chunks (one chunk per "##" section;
     the FAQ file gets one chunk per question).
  2. Turns each chunk into an embedding with a local sentence-transformers model.
  3. Saves chunks + embeddings in kb_chunks (old rows are replaced, so you can
     re-run it after editing a policy file).
  4. Runs one test search so you can see retrieval working.

Everything is saved in ONE transaction: if anything fails, nothing changes.
"""

import getpass
import re
import sys
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384  # must match vector(384) in kb_schema.sql
# bge models work best when SEARCH QUERIES (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TEST_QUESTION = "How long does a credit card refund take?"


# ---------------------------------------------------------------- chunking
def parse_front_matter(text):
    """Split the '--- doc_id / title / category ---' header from the body."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, m.group(2)


def split_sections(body):
    """Return a list of (section_name, text) from the markdown body."""
    lines = body.strip().split("\n")
    if lines and lines[0].startswith("# "):  # drop the H1 title line
        lines = lines[1:]

    sections, name, current = [], "Overview", []
    for line in lines:
        if line.startswith("## "):
            sections.append((name, "\n".join(current).strip()))
            name, current = line[3:].strip(), []
        else:
            current.append(line)
    sections.append((name, "\n".join(current).strip()))
    sections = [(n, t) for n, t in sections if t]

    # A file with no "##" headings at all (the FAQ): one chunk per paragraph
    if not any(line.startswith("## ") for line in lines):
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            sections = [(re.sub(r"\*", "", p.split("\n")[0])[:80], p) for p in paragraphs]
    return sections


def build_chunks(folder):
    chunks = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        meta, body = parse_front_matter(text)
        doc_id = meta.get("doc_id", path.stem)
        title = meta.get("title", path.stem)
        category = meta.get("category", "general")

        seen = {}
        for name, content in split_sections(body):
            seen[name] = seen.get(name, 0) + 1
            section = name if seen[name] == 1 else f"{name} ({seen[name]})"
            chunks.append({
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "section": section,
                # the title is part of the text so each chunk makes sense alone
                "chunk_text": f"{title} - {section}\n{content}",
            })
    return chunks


# ---------------------------------------------------------------- loading
def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def main():
    import psycopg
    from sentence_transformers import SentenceTransformer

    arg = sys.argv[1] if len(sys.argv) > 1 else input("Folder with the .md policy files: ")
    folder = Path(arg.strip().strip('"'))
    if not any(folder.glob("*.md")) and (folder / "kb").is_dir():
        folder = folder / "kb"
    chunks = build_chunks(folder)
    if not chunks:
        sys.exit(f"No .md files found in {folder}")
    n_docs = len({c["doc_id"] for c in chunks})
    print(f"{len(chunks)} chunks from {n_docs} files in {folder}\n")

    print("Loading the embedding model (first run downloads ~130 MB)...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    if dim != EXPECTED_DIM:
        sys.exit(f"Model gives {dim} dimensions but kb_schema.sql expects {EXPECTED_DIM}.")

    for c in chunks:  # warn if a chunk is longer than the model can read
        n_tokens = len(model.tokenizer.encode(c["chunk_text"]))
        if n_tokens > model.max_seq_length:
            print(f"  WARNING: {c['doc_id']} / {c['section']} has {n_tokens} tokens "
                  f"(model reads {model.max_seq_length}); the end is ignored.")

    vectors = model.encode([c["chunk_text"] for c in chunks],
                           normalize_embeddings=True, show_progress_bar=True)

    host = input("\nRDS endpoint: ").strip()
    dbname = input("Database name [postgres]: ").strip() or "postgres"
    user = input("Master username [postgres]: ").strip() or "postgres"
    password = getpass.getpass("Password (hidden): ")

    with psycopg.connect(host=host, port=5432, dbname=dbname,
                         user=user, password=password, sslmode="require") as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE kb_chunks RESTART IDENTITY")
            for c, v in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO kb_chunks (doc_id, title, category, section, chunk_text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                    (c["doc_id"], c["title"], c["category"], c["section"],
                     c["chunk_text"], to_pgvector(v)))
            print(f"\nSaved {len(chunks)} chunks.")

            # Test search: the nearest chunks to a question
            q = model.encode(QUERY_PREFIX + TEST_QUESTION, normalize_embeddings=True)
            qv = to_pgvector(q)
            cur.execute(
                "SELECT doc_id, section, 1 - (embedding <=> %s::vector) AS score "
                "FROM kb_chunks ORDER BY embedding <=> %s::vector LIMIT 3", (qv, qv))
            print(f'\nTest search: "{TEST_QUESTION}"')
            for doc_id, section, score in cur.fetchall():
                print(f"  {score:.3f}  {doc_id}  {section}")
    print("\nDone.")


if __name__ == "__main__":
    import psycopg
    try:
        main()
    except psycopg.errors.UndefinedTable:
        sys.exit("\nTable kb_chunks does not exist. Run kb_schema.sql first (as the admin user).")
    except psycopg.OperationalError as e:
        sys.exit("\nCould not connect. Check endpoint, database name, password, and that your "
                 "CURRENT IP is allowed on port 5432 in the security group.\n" + str(e))
