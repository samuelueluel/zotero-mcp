"""NUL characters must never reach Chroma's full-text index."""

import sqlite3

import chromadb

from zotero_mcp.chroma_client import ChromaClient


def _fts_ok(path) -> str:
    con = sqlite3.connect(f"file:{path}/chroma.sqlite3?mode=ro", uri=True)
    try:
        return con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()


def test_fts_safe_drops_nul_only():
    assert ChromaClient._fts_safe(["Drut\x00a, O.", "tab\tok", None]) == ["Druta, O.", "tab\tok", None]


def test_guarded_text_keeps_the_fts_index_valid(tmp_path):
    # With chromadb 1.x a raw NUL here makes PRAGMA integrity_check report
    # "malformed inverted index for FTS5 table main.embedding_fulltext_search".
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("nul-test").add(
        ids=["a"], documents=ChromaClient._fts_safe(["Seymour, E., & Drut\x00a, O."]), embeddings=[[0.1, 0.2]])
    del client
    assert _fts_ok(tmp_path) == "ok"
