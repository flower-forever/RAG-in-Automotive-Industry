import os
import pytest
from src.ingestion.csaf_parser import parse_single_csaf, parse_all_csaf_dir

# Path to the sample CSAF JSON file in the project
SAMPLE_FILE = "doc/cisa_csaf/icsa-26-148-01.json"
CSAF_DIR = "doc/cisa_csaf"

def test_parse_single_csaf_exists():
    """Verify that sample file exists and is parsable."""
    assert os.path.exists(SAMPLE_FILE), f"Sample file {SAMPLE_FILE} does not exist"
    
def test_parse_single_csaf_chunks():
    """Verify the chunking structure of a parsed CSAF file."""
    chunks = parse_single_csaf(SAMPLE_FILE)
    
    assert len(chunks) > 0, "No chunks parsed from CSAF file"
    
    # Verify overview chunk exists
    overview_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "overview"]
    assert len(overview_chunks) == 1, "There should be exactly one overview chunk per CSAF file"
    
    overview = overview_chunks[0]
    assert overview["metadata"]["source"] == "CISA_CSAF"
    assert "Advisory ID" in overview["text"]
    assert "Advisory Summary" in overview["text"]
    
    # Verify vulnerability chunks exist
    vuln_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "vulnerability"]
    assert len(vuln_chunks) > 0, "There should be vulnerability chunks"
    
    # Check vulnerability metadata and text
    first_vuln = vuln_chunks[0]
    assert "cve" in first_vuln["metadata"]
    assert first_vuln["metadata"]["cve"].startswith("CVE-")
    assert "cwe_id" in first_vuln["metadata"]
    assert "Vulnerability CVE" in first_vuln["text"]
    assert "Description" in first_vuln["text"]
    
    # Verify recommended practice chunks exist (Fine-grained)
    rec_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "recommendation"]
    assert len(rec_chunks) > 0, "There should be recommendation chunks"
    assert "Recommendation" in rec_chunks[0]["text"]
    assert rec_chunks[0]["metadata"]["chunk_type"] == "recommendation"

def test_parse_all_csaf_dir():
    """Verify that all JSON files in the CSAF directory parse without errors."""
    assert os.path.isdir(CSAF_DIR), f"Directory {CSAF_DIR} does not exist"
    
    # Let's count JSON files
    json_files = [f for f in os.listdir(CSAF_DIR) if f.endswith(".json")]
    assert len(json_files) == 200, f"Expected 200 files, found {len(json_files)}"
    
    chunks = parse_all_csaf_dir(CSAF_DIR)
    assert len(chunks) > len(json_files), "Total chunk count should exceed file count due to splitting"
    
    # Ensure all chunks have the required basic structure
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert isinstance(chunk["text"], str)
        assert isinstance(chunk["metadata"], dict)
        assert chunk["metadata"]["source"] == "CISA_CSAF"
        assert "advisory_id" in chunk["metadata"]
        assert "chunk_type" in chunk["metadata"]
        assert chunk["metadata"]["chunk_type"] in ["overview", "vulnerability", "recommendation"]
