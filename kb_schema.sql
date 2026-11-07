-- Policy knowledge base for RAG.
-- Run ONCE as the admin (master) user, in the same database as your other tables.
-- Embedding model: BAAI/bge-small-en-v1.5  ->  384 dimensions.
-- If you ever switch to a model with a different size, change 384 and re-run load_kb.py.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE kb_chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    doc_id     TEXT NOT NULL,              -- kb-01 ... kb-13
    title      TEXT NOT NULL,              -- title of the policy file
    category   TEXT NOT NULL,              -- refunds, delivery, returns, ...
    section    TEXT NOT NULL,              -- the "##" heading (or the FAQ question)
    chunk_text TEXT NOT NULL,              -- title + section + text (what the LLM reads)
    embedding  vector(384) NOT NULL,
    -- generated automatically; lets you try keyword search + vector search together later
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    UNIQUE (doc_id, section)
);

CREATE INDEX idx_kb_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_kb_tsv       ON kb_chunks USING gin (tsv);