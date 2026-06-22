import os
import pytest
from unittest.mock import MagicMock, patch
from src.generation import format_retrieved_chunks, sigmoid, SecureOpsGenerator

@pytest.fixture
def mock_chunks():
    return [
        {
            "id": "doc_0",
            "text": "The Danelec MacGregor VDR G4e suffers from default credentials.",
            "rerank_score": 2.5,
            "metadata": {
                "source": "CISA_CSAF",
                "advisory_id": "ICSA-26-148-01",
                "vendor": "Danelec",
                "products": "MacGregor VDR G4e",
                "date": "2026-05-28",
                "severity": "HIGH",
                "chunk_type": "vulnerability"
            }
        },
        {
            "id": "doc_1",
            "text": "NIST recommends network segmentation for industrial control systems.",
            "rerank_score": 0.5,
            "metadata": {
                "source": "NIST_SP_800-82_R3",
                "chapter": "6. Network Architecture",
                "section": "6.2 Network Segmentation",
                "page_start": 145,
                "page_end": 145
            }
        }
    ]

def test_sigmoid():
    """Verify that sigmoid maps float values correctly to a [0, 1] range."""
    assert sigmoid(0.0) == 0.5
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01

def test_format_retrieved_chunks(mock_chunks):
    """Verify formatting of CSAF and PDF metadata headers in the context block."""
    context_text = format_retrieved_chunks(mock_chunks)
    
    # Assert CSAF format
    assert "CISA ICS Advisory ICSA-26-148-01" in context_text
    assert "Vendor: Danelec" in context_text
    
    # Assert PDF format
    assert "Source: NIST_SP_800-82_R3" in context_text
    assert "Chapter: 6. Network Architecture" in context_text
    assert "Pages: 145-145" in context_text

def test_empty_retrieval_rejection():
    """Verify that empty retrieval returns an immediate honest rejection."""
    generator = SecureOpsGenerator(api_key="mock_key")
    answer, confidence, cited = generator.generate_answer("How to configure firewall?", [])
    
    assert "I don't have enough information" in answer
    assert confidence == 0.0
    assert cited == []

def test_low_confidence_rejection(mock_chunks):
    """Verify that low rerank scores trigger the honest rejection threshold."""
    # Set top score below threshold (-3.0)
    mock_chunks[0]["rerank_score"] = -4.5
    
    generator = SecureOpsGenerator(api_key="mock_key", low_confidence_threshold=-3.0)
    answer, confidence, cited = generator.generate_answer("How to configure firewall?", mock_chunks)
    
    assert "I don't have enough information" in answer
    # Confidence should be low (sigmoid of -4.5 is ~0.01)
    assert confidence < 0.02
    assert cited == []

@patch("google.generativeai.GenerativeModel")
def test_generation_with_api_key(mock_model_class, mock_chunks):
    """Verify generator behavior when an API key is provided, using mocked API responses with CoT tags."""
    # Set up mock response
    mock_response = MagicMock()
    mock_response.text = "<thinking>\nAnalyzing the context...\n</thinking>\n<answer>\nThe Danelec VDR G4e has default credentials [Index 1]. Learn more in [Index 2].\n\nSources Cited:\n[1] Danelec Advisory\n</answer>"
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance
    
    generator = SecureOpsGenerator(api_key="mock_api_key")
    assert generator._has_client is True
    
    answer, confidence, cited = generator.generate_answer("What vulnerability affects Danelec?", mock_chunks)
    
    # Assert correct response text and parsed citations. The <thinking> block should be stripped.
    assert "The Danelec VDR G4e" in answer
    assert "<thinking>" not in answer
    assert confidence == sigmoid(2.5)  # based on top chunk rerank_score (2.5)
    
    # We cited [Index 1] and [Index 2] in the mocked response
    assert len(cited) == 2
    assert cited[0]["id"] == "doc_0"
    assert cited[1]["id"] == "doc_1"

@patch("google.generativeai.GenerativeModel")
def test_cot_rejection(mock_model_class, mock_chunks):
    """Verify that if the LLM decides to refuse inside the thinking block, we return the refusal cleanly."""
    mock_response = MagicMock()
    # No <answer> block, just a direct refusal or an answer block containing a refusal
    mock_response.text = "<thinking>\nThe user is asking about our internal company firewall. I must refuse.</thinking>\n<answer>\nI don't have enough information in my knowledge base to answer this.</answer>"
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance
    
    generator = SecureOpsGenerator(api_key="mock_api_key")
    answer, confidence, cited = generator.generate_answer("What is our company's firewall config?", mock_chunks)
    
    assert "I don't have enough information" in answer
    assert cited == []

@patch("google.generativeai.GenerativeModel")
def test_critique_pass(mock_model_class, mock_chunks):
    """Verify that a PASS critique allows the answer to be returned."""
    # Mock critique response
    mock_critique_response = MagicMock()
    mock_critique_response.text = "PASS"
    
    # Mock main response
    mock_main_response = MagicMock()
    mock_main_response.text = "<thinking>Analysis</thinking><answer>The Danelec VDR G4e has default credentials [Index 1].</answer>"
    
    mock_model_instance = MagicMock()
    # The first call is main generation, second is critique
    mock_model_instance.generate_content.side_effect = [mock_main_response, mock_critique_response]
    mock_model_class.return_value = mock_model_instance
    
    generator = SecureOpsGenerator(api_key="mock_api_key")
    answer, confidence, cited = generator.generate_answer("What vulnerability affects Danelec?", mock_chunks)
    
    assert "The Danelec VDR G4e" in answer
    assert len(cited) == 1

@patch("google.generativeai.GenerativeModel")
def test_critique_fail(mock_model_class, mock_chunks):
    """Verify that a FAIL critique triggers a refusal."""
    mock_critique_response = MagicMock()
    mock_critique_response.text = "FAIL - The answer contains unsupported claims."
    
    mock_main_response = MagicMock()
    mock_main_response.text = "<thinking>Analysis</thinking><answer>Danelec VDR G4e was hacked by APT41 [Index 1].</answer>"
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.side_effect = [mock_main_response, mock_critique_response]
    mock_model_class.return_value = mock_model_instance
    
    generator = SecureOpsGenerator(api_key="mock_api_key")
    answer, confidence, cited = generator.generate_answer("Who hacked Danelec?", mock_chunks)
    
    assert "I don't have enough information" in answer
    assert cited == []

@patch.dict(os.environ, {}, clear=True)
def test_generation_no_api_key_warning(mock_chunks):
    """Verify that if no API key is set, the generator returns a warning note but does not crash."""
    generator = SecureOpsGenerator(api_key=None)
    # Ensure client configuration was skipped
    assert generator._has_client is False
    
    answer, confidence, cited = generator.generate_answer("What vulnerability affects Danelec?", mock_chunks)
    
    assert "Gemini API Key not configured" in answer
    assert confidence == sigmoid(2.5)
    assert len(cited) == 2  # returns candidates as placeholder

@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="Requires GEMINI_API_KEY environment variable")
def test_live_generation_if_key_present(mock_chunks):
    """Integrity test for live API connection, executed only if key is configured."""
    generator = SecureOpsGenerator()
    answer, confidence, cited = generator.generate_answer(
        "Is there default credentials in Daneregor G4e devices?",
        mock_chunks
    )
    
    assert isinstance(answer, str)
    assert len(answer) > 10
    assert confidence > 0.0
