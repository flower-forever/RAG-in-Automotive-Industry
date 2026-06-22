import os
import shutil
import pytest
from src.indexing import build_index
from src.retrieval import SecureOpsRetriever

CSAF_DIR = "doc/cisa_csaf"
CSF_PDF = "doc/NIST Cybersecurity Framework(CSF) 2.0.pdf"
NIST_PDF = "doc/NIST.SP.800-82r3.pdf"
TEST_DB_PATH = "tests/temp_test_db"
TEST_CSAF_DIR = "tests/temp_csaf"

@pytest.fixture(scope="module", autouse=True)
def setup_test_index():
    """Build a temporary index before running module tests."""
    # Clean folders on startup
    for path in [TEST_DB_PATH, TEST_CSAF_DIR]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception:
                pass
                
    # Create temp CSAF folder with 2 files for fast indexing
    os.makedirs(TEST_CSAF_DIR, exist_ok=True)
    sample_files = ["icsa-26-148-01.json", "icsa-26-008-01.json"]
    for f_name in sample_files:
        src = os.path.join(CSAF_DIR, f_name)
        dst = os.path.join(TEST_CSAF_DIR, f_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    # Index pages
    build_index(
        csaf_dir=TEST_CSAF_DIR,
        csf_pdf_path=CSF_PDF,
        nist_pdf_path=NIST_PDF,
        db_path=TEST_DB_PATH,
        collection_name="test_collection",
        limit_pdf_pages=True
    )
    
    yield
    
    # Teardown: Clean up CSAF directory. DB path is locked until pytest exits.
    if os.path.exists(TEST_CSAF_DIR):
        try:
            shutil.rmtree(TEST_CSAF_DIR)
        except Exception:
            pass

@pytest.fixture
def retriever():
    """Helper fixture to load retriever initialized on test database."""
    return SecureOpsRetriever(
        db_path=TEST_DB_PATH,
        collection_name="test_collection"
    )

def test_dense_search(retriever):
    """Verify that dense search returns candidates with metadata and scores."""
    hits = retriever.dense_search("vulnerability remote access security", top_k=5)
    assert len(hits) > 0
    assert "text" in hits[0]
    assert "metadata" in hits[0]
    assert "score" in hits[0]
    # Similarity score should be bounded
    assert -1.0 <= hits[0]["score"] <= 1.0

def test_sparse_search(retriever):
    """Verify that sparse search matching specific terms works."""
    # Search for a term that appears in CSAF JSON files
    hits = retriever.sparse_search("CVE-2026-42941", top_k=5)
    assert len(hits) > 0
    # The hit should match the exact CVE
    assert "CVE-2026-42941" in hits[0]["text"]
    assert hits[0]["metadata"]["cve"] == "CVE-2026-42941"

def test_rrf_fusion(retriever):
    """Verify that Reciprocal Rank Fusion combines dense and sparse hits."""
    dense_hits = retriever.dense_search("Danelec default credentials", top_k=5)
    sparse_hits = retriever.sparse_search("CVE-2026-42941", top_k=5)
    
    fused = retriever.reciprocal_rank_fusion(dense_hits, sparse_hits)
    assert len(fused) > 0
    assert "rrf_score" in fused[0]
    # RRF score should be greater than 0
    assert fused[0]["rrf_score"] > 0.0

def test_cross_encoder_rerank(retriever):
    """Verify that retrieve method uses Cross-Encoder and returns ranked top k chunks."""
    results = retriever.retrieve("How to protect OT remote access?", k=3)
    assert len(results) > 0
    assert len(results) <= 3
    assert "rerank_score" in results[0]
    # Verify that results are sorted in descending order of rerank score
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_metadata_filtering(retriever):
    """Verify that metadata filtering matches exact specifications for retrieval."""
    # Filter by vendor
    results = retriever.retrieve("vulnerability fix", k=5, vendor="Danelec")
    for r in results:
        # Check that vendor matches
        assert "Danelec" in r["metadata"]["vendor"]
        
    # Filter by severity
    results = retriever.retrieve("vulnerability", k=5, severity="HIGH")
    for r in results:
        assert r["metadata"]["severity"] == "HIGH"
        
    # Filter by source
    results = retriever.retrieve("OT security", k=5, source="NIST_CSF_2.0")
    for r in results:
        assert r["metadata"]["source"] == "NIST_CSF_2.0"

def test_multi_query_retrieve(retriever):
    """Verify that retrieval accepts a list of queries, pools them, and returns properly reranked results."""
    queries = [
        "How to protect OT remote access?",
        "What are ICS remote connectivity guidelines?"
    ]
    results = retriever.retrieve(queries, k=3)
    assert len(results) > 0
    assert len(results) <= 3
    assert "rerank_score" in results[0]
    # Verify that results are sorted in descending order of rerank score
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
