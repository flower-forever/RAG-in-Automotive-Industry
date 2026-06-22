import os
import re
import pickle
import pytest
from unittest.mock import MagicMock, patch, ANY
from src.cli import SecureOpsCLI

def strip_ansi(text: str) -> str:
    """Helper to remove ANSI escape sequences from console output."""
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

@pytest.fixture
def mock_cli():
    # Patch check_database_status to prevent print during init
    with patch.object(SecureOpsCLI, 'check_database_status'):
        cli = SecureOpsCLI(
            db_path="mock_db_path",
            csaf_dir="mock_csaf_dir",
            csf_pdf_path="mock_csf_path",
            nist_pdf_path="mock_nist_path"
        )
        yield cli

def test_cli_initial_state(mock_cli):
    """Verify that CLI initializes with empty filters and correct prompt."""
    assert mock_cli.filters["vendor"] is None
    assert mock_cli.filters["severity"] is None
    assert mock_cli.filters["source"] is None
    assert mock_cli.query_history == []
    assert "SecureOps>" in strip_ansi(mock_cli.prompt)

def test_cli_emptyline(mock_cli, capsys):
    """Verify that an empty line command does nothing and prints nothing."""
    mock_cli.emptyline()
    captured = capsys.readouterr().out
    assert captured == ""

def test_cli_filters(mock_cli, capsys):
    """Verify that metadata filters are set, shown, and cleared correctly."""
    # 1. Show empty filters
    mock_cli.onecmd("filter")
    captured = strip_ansi(capsys.readouterr().out)
    assert "Active Filters:" in captured
    assert "No active filters." in captured

    # 2. Set vendor filter
    mock_cli.onecmd("filter vendor Siemens")
    captured = strip_ansi(capsys.readouterr().out)
    assert "Filter for vendor set to 'Siemens'" in captured
    assert mock_cli.filters["vendor"] == "Siemens"
    assert "Siemens" in strip_ansi(mock_cli.prompt)

    # 3. Set severity filter
    mock_cli.onecmd("filter severity HIGH")
    captured = strip_ansi(capsys.readouterr().out)
    assert "Filter for severity set to 'HIGH'" in captured
    assert mock_cli.filters["severity"] == "HIGH"
    assert "HIGH" in strip_ansi(mock_cli.prompt)

    # 4. View active filters
    mock_cli.onecmd("filter")
    captured = strip_ansi(capsys.readouterr().out)
    assert "vendor: Siemens" in captured
    assert "severity: HIGH" in captured

    # 5. Filter clear
    mock_cli.onecmd("filter clear")
    captured = strip_ansi(capsys.readouterr().out)
    assert "All filters cleared" in captured
    assert mock_cli.filters["vendor"] is None
    assert mock_cli.filters["severity"] is None
    assert "SecureOps>" in strip_ansi(mock_cli.prompt)

    # 6. Invalid filter command
    mock_cli.onecmd("filter invalid_key value")
    captured = strip_ansi(capsys.readouterr().out)
    assert "Unknown filter type" in captured

def test_cli_history(mock_cli, capsys):
    """Verify that history lists previous queries."""
    # Empty history
    mock_cli.onecmd("history")
    captured = strip_ansi(capsys.readouterr().out)
    assert "No queries in history yet" in captured

    # Add queries manually and print
    mock_cli.query_history = ["What is CSF?", "How to patch Danelec?"]
    mock_cli.onecmd("history")
    captured = strip_ansi(capsys.readouterr().out)
    assert "Query History:" in captured
    assert "1. What is CSF?" in captured
    assert "2. How to patch Danelec?" in captured

def test_cli_exit(mock_cli, capsys):
    """Verify that exit commands return True to stop the loop."""
    assert mock_cli.onecmd("exit") is True
    assert mock_cli.onecmd("quit") is True
    assert mock_cli.onecmd("EOF") is True
    captured = strip_ansi(capsys.readouterr().out)
    assert "Goodbye!" in captured

@patch("builtins.open")
@patch("os.path.exists")
@patch("pickle.load")
def test_cli_sources(mock_pickle_load, mock_exists, mock_open, mock_cli, capsys):
    """Verify that the sources command correctly parses and displays BM25 statistics."""
    mock_exists.return_value = True
    
    # Mock data inside the BM25 index file
    mock_pickle_load.return_value = {
        "metadatas": [
            {"source": "CISA_CSAF", "vendor": "Siemens"},
            {"source": "CISA_CSAF", "vendor": "Siemens"},
            {"source": "CISA_CSAF", "vendor": "Schneider Electric"},
            {"source": "NIST_SP_800-82_R3"},
            {"source": "NIST_CSF_2.0"}
        ]
    }
    
    mock_cli.onecmd("sources")
    captured = strip_ansi(capsys.readouterr().out)
    
    assert "Index Data Summary:" in captured
    assert "Total Chunks Indexed: 5" in captured
    assert "CISA_CSAF: 3 chunks" in captured
    assert "NIST_SP_800-82_R3: 1 chunks" in captured
    assert "NIST_CSF_2.0: 1 chunks" in captured
    assert "Siemens: 2 chunks" in captured
    assert "Schneider Electric: 1 chunks" in captured

def test_cli_ask_empty(mock_cli, capsys):
    """Verify that ask with empty query returns an error."""
    mock_cli.onecmd("ask   ")
    captured = strip_ansi(capsys.readouterr().out)
    assert "Please provide a question" in captured

@patch.object(SecureOpsCLI, "ensure_models_loaded")
def test_cli_ask_flow(mock_ensure, mock_cli, capsys):
    """Verify that the ask command calls retriever/generator and displays formatted results."""
    mock_ensure.return_value = True
    
    # Mock retriever and generator instances
    mock_cli.retriever = MagicMock()
    mock_cli.generator = MagicMock()
    
    # Setup mocks
    mock_chunks = [
        {
            "id": "doc_0",
            "text": "Danelec vulnerability details.",
            "metadata": {
                "source": "CISA_CSAF",
                "advisory_id": "ICSA-26-148-01",
                "vendor": "Danelec",
                "severity": "CRITICAL"
            }
        }
    ]
    mock_cli.retriever.retrieve.return_value = mock_chunks
    
    mock_cli.generator.generate_answer.return_value = (
        "This is the generated answer [Index 1].",
        0.85,
        mock_chunks
    )
    
    # Run command
    mock_cli.onecmd("ask What is the Danelec vulnerability?")
    captured = strip_ansi(capsys.readouterr().out)
    
    # Verify mock calls
    mock_cli.retriever.retrieve.assert_called_once_with(
        query_or_queries=ANY,
        k=5,
        vendor=None,
        severity=None,
        source=None
    )
    mock_cli.generator.generate_answer.assert_called_once_with(
        query="What is the Danelec vulnerability?",
        retrieved_chunks=mock_chunks
    )
    
    # Verify stdout outputs
    assert "Question: What is the Danelec vulnerability?" in captured
    assert "Answer:" in captured
    assert "This is the generated answer [Index 1]." in captured
    assert "Retrieval Confidence: 85.0%" in captured
    assert "Sources Details:" in captured
    assert "Source: CISA_CSAF" in captured
    assert "Advisory ID: ICSA-26-148-01" in captured
    assert "Vendor: Danelec" in captured
    assert "Severity: CRITICAL" in captured

@patch("src.cli.build_index")
@patch("builtins.input")
def test_cli_rebuild_confirm(mock_input, mock_build, mock_cli, capsys):
    """Verify that rebuild runs index building when user confirms."""
    mock_input.return_value = "y"
    mock_build.return_value = (10, 10)
    
    mock_cli.onecmd("rebuild")
    captured = strip_ansi(capsys.readouterr().out)
    
    mock_build.assert_called_once_with(
        csaf_dir="mock_csaf_dir",
        csf_pdf_path="mock_csf_path",
        nist_pdf_path="mock_nist_path",
        db_path="mock_db_path",
        collection_name="secureops_assistant",
        limit_pdf_pages=False
    )
    assert "Database rebuilt successfully" in captured
    # Retriever and generator should be cleared to trigger reloading on next query
    assert mock_cli.retriever is None
    assert mock_cli.generator is None

@patch("src.cli.build_index")
@patch("builtins.input")
def test_cli_rebuild_cancel(mock_input, mock_build, mock_cli, capsys):
    """Verify that rebuild is canceled when user rejects confirmation."""
    mock_input.return_value = "n"
    
    mock_cli.onecmd("rebuild")
    captured = strip_ansi(capsys.readouterr().out)
    
    mock_build.assert_not_called()
    assert "Rebuild canceled." in captured
