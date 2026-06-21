import os
import shutil
import pytest
import pickle
import chromadb
from src.indexing import tokenize_text, build_index

CSAF_DIR = "doc/cisa_csaf"
CSF_PDF = "doc/NIST Cybersecurity Framework(CSF) 2.0.pdf"
NIST_PDF = "doc/NIST.SP.800-82r3.pdf"
TEST_DB_PATH = "tests/temp_test_db"
TEST_CSAF_DIR = "tests/temp_csaf"

@pytest.fixture(autouse=True)
def cleanup_temp_db():
    """Fixture to ensure the temp test database folder is cleaned up before and after tests."""
    for path in [TEST_DB_PATH, TEST_CSAF_DIR]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception:
                pass
    yield
    # Clean up the CSAF folder since it's not locked.
    # DB path is locked by ChromaDB SQLite on Windows until process exits,
    # so we leave it to be cleaned up on the next run's startup.
    if os.path.exists(TEST_CSAF_DIR):
        try:
            shutil.rmtree(TEST_CSAF_DIR)
        except Exception:
            pass

def test_tokenize_text():
    """Verify that specialized security tokenization preserves hyphens, dots, and case-insensitivity."""
    text = "Find CVE-2026-42941 and subcategory PR.AC-1 inside NIST.SP.800-82r3."
    tokens = tokenize_text(text)
    
    # Verify exact preservation of custom terms
    assert "cve-2026-42941" in tokens
    assert "pr.ac-1" in tokens
    assert "nist.sp.800-82r3." not in tokens # should strip final punctuation but keep inner dots
    assert "nist.sp.800-82r3" in tokens
    assert "find" in tokens

def test_build_index_partial():
    """Verify that indexing works correctly on a partial set of pages."""
    # Create temp CSAF directory with only 2 files for fast testing
    os.makedirs(TEST_CSAF_DIR, exist_ok=True)
    sample_files = ["icsa-26-148-01.json", "icsa-26-008-01.json"]
    for f_name in sample_files:
        src = os.path.join(CSAF_DIR, f_name)
        dst = os.path.join(TEST_CSAF_DIR, f_name)
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Build a small index with limited pages for speed in tests
    num_docs, num_chunks = build_index(
        csaf_dir=TEST_CSAF_DIR,
        csf_pdf_path=CSF_PDF,
        nist_pdf_path=NIST_PDF,
        db_path=TEST_DB_PATH,
        collection_name="test_collection",
        limit_pdf_pages=True  # extremely important: parses only 5+6 pages of PDFs for speed
    )
    
    assert num_chunks > 0, "No chunks indexed"
    
    # 1. Verify ChromaDB persistence
    assert os.path.exists(TEST_DB_PATH), "Database path was not created"
    
    client = chromadb.PersistentClient(path=TEST_DB_PATH)
    collection = client.get_collection(name="test_collection")
    assert collection.count() == num_chunks, "ChromaDB count doesn't match total indexed chunks"
    
    # Verify metadata-based queries in ChromaDB
    res = collection.get(where={"source": "CISA_CSAF"}, limit=5)
    assert len(res["ids"]) > 0, "Could not retrieve CSAF documents from ChromaDB"
    
    # 2. Verify BM25 serialization
    bm25_file = os.path.join(TEST_DB_PATH, "bm25_index.pkl")
    assert os.path.exists(bm25_file), "BM25 index pickle file was not created"
    
    with open(bm25_file, "rb") as f:
        bm25_data = pickle.load(f)
        
    assert "bm25" in bm25_data
    assert "texts" in bm25_data
    assert len(bm25_data["texts"]) == num_chunks
    assert len(bm25_data["metadatas"]) == num_chunks
    
    # Check that BM25 can run a search query
    query_tokens = tokenize_text("CVE-2026-42941")
    scores = bm25_data["bm25"].get_scores(query_tokens)
    assert len(scores) == num_chunks
