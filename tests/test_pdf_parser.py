import os
import pytest
from src.ingestion.pdf_parser import clean_title, update_header_path, parse_pdf_to_chunks

CSF_PDF = "doc/NIST Cybersecurity Framework(CSF) 2.0.pdf"
NIST_PDF = "doc/NIST.SP.800-82r3.pdf"

def test_clean_title():
    """Verify that formatting symbols are stripped from headers."""
    assert clean_title("**5.1. Title**") == "5.1. Title"
    assert clean_title("_Introduction_") == "Introduction"
    assert clean_title("`Code Section` ") == "Code Section"

def test_update_header_path():
    """Verify that hierarchical heading tracking updates correctly."""
    path = []
    
    # Encounter Level 1 heading
    path = update_header_path(path, 1, "Chapter 1")
    assert path == ["Chapter 1"]
    
    # Encounter Level 2 heading
    path = update_header_path(path, 2, "Section 1.1")
    assert path == ["Chapter 1", "Section 1.1"]
    
    # Encounter Level 3 heading
    path = update_header_path(path, 3, "Subsection 1.1.1")
    assert path == ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
    
    # Encounter Level 2 heading (sibling)
    path = update_header_path(path, 2, "Section 1.2")
    assert path == ["Chapter 1", "Section 1.2"]
    
    # Encounter Level 1 heading (sibling)
    path = update_header_path(path, 1, "Chapter 2")
    assert path == ["Chapter 2"]

def test_parse_csf_pdf_chunks():
    """Verify parsing a page range of the CSF 2.0 PDF works and retains structure."""
    assert os.path.exists(CSF_PDF), f"File {CSF_PDF} does not exist"
    
    # Parse pages 13-14 (0-indexed pages 13-14 corresponds to physical pages 14-15 which we inspected)
    # PyMuPDF pages argument is 0-indexed list of page indices
    chunks = parse_pdf_to_chunks(CSF_PDF, "NIST_CSF_2.0", pages=[13, 14], max_words=300)
    
    assert len(chunks) > 0, "No chunks generated from CSF PDF"
    
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        metadata = chunk["metadata"]
        assert metadata["source"] == "NIST_CSF_2.0"
        assert "chapter" in metadata
        assert "page_start" in metadata
        assert "page_end" in metadata
        assert metadata["page_start"] in [14, 15]
        
        # Verify prepended context
        assert "Source: NIST_CSF_2.0" in chunk["text"]
        assert "Pages:" in chunk["text"]

def test_parse_nist_pdf_chunks():
    """Verify parsing a small range of NIST SP 800-82 works."""
    assert os.path.exists(NIST_PDF), f"File {NIST_PDF} does not exist"
    
    # Parse pages 145-146 (corresponds to network architectures section)
    chunks = parse_pdf_to_chunks(NIST_PDF, "NIST_SP_800-82_R3", pages=[144, 145], max_words=300)
    
    assert len(chunks) > 0, "No chunks generated from NIST SP 800-82 PDF"
    
    # Verify metadata fields are preserved
    first_chunk = chunks[0]
    assert first_chunk["metadata"]["source"] == "NIST_SP_800-82_R3"
    assert first_chunk["metadata"]["page_start"] in [145, 146]
