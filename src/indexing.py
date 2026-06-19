import os
import re
import pickle
import glob
from typing import List, Dict, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from src.ingestion.csaf_parser import parse_all_csaf_dir
from src.ingestion.pdf_parser import parse_pdf_to_chunks

# Special tokenizer for BM25 to preserve security terms (CVEs, PR.AC codes, etc.)
def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text by converting to lowercase and extracting words including hyphens and periods.
    Preserves terms like 'cve-2026-42941', 'nist.sp.800-82r3', 'pr.ac'
    """
    return re.findall(r'[a-zA-Z0-9]+(?:[\-\.][a-zA-Z0-9]+)*', text.lower())

def get_embedding_model(model_name: str = "BAAI/bge-base-en-v1.5") -> SentenceTransformer:
    """
    Load the SentenceTransformer model.
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    return model

def build_index(
    csaf_dir: str,
    csf_pdf_path: str,
    nist_pdf_path: str,
    db_path: str = "chroma_db",
    collection_name: str = "secureops_assistant",
    model_name: str = "BAAI/bge-base-en-v1.5",
    limit_pdf_pages: bool = False
) -> Tuple[int, int]:
    """
    Main function to parse all documents, embed them with BGE, 
    and save them to ChromaDB and BM25 index.
    Returns (num_documents, num_chunks).
    """
    # 1. Parse all document sources to collect chunks
    all_chunks = []
    
    # Ingest CSAF JSONs
    print("Ingesting CSAF JSON files...")
    if os.path.exists(csaf_dir):
        csaf_chunks = parse_all_csaf_dir(csaf_dir)
        all_chunks.extend(csaf_chunks)
        print(f"CSAF Ingestion complete. Created {len(csaf_chunks)} chunks.")
    else:
        print(f"CSAF directory not found: {csaf_dir}")
        
    # Ingest CSF 2.0 PDF
    print("Ingesting NIST CSF 2.0 PDF...")
    if os.path.exists(csf_pdf_path):
        # If limiting pages (e.g. for fast tests), parse pages 0-4. Otherwise parse all pages.
        pages = [i for i in range(5)] if limit_pdf_pages else None
        csf_chunks = parse_pdf_to_chunks(csf_pdf_path, "NIST_CSF_2.0", pages=pages)
        all_chunks.extend(csf_chunks)
        print(f"CSF PDF Ingestion complete. Created {len(csf_chunks)} chunks.")
    else:
        print(f"CSF PDF path not found: {csf_pdf_path}")
        
    # Ingest NIST SP 800-82 PDF
    print("Ingesting NIST SP 800-82 Rev 3 PDF...")
    if os.path.exists(nist_pdf_path):
        # If limiting pages (e.g. for fast tests), parse pages 140-145. Otherwise parse all.
        pages = [i for i in range(140, 146)] if limit_pdf_pages else None
        nist_chunks = parse_pdf_to_chunks(nist_pdf_path, "NIST_SP_800-82_R3", pages=pages)
        all_chunks.extend(nist_chunks)
        print(f"NIST SP 800-82 Ingestion complete. Created {len(nist_chunks)} chunks.")
    else:
        print(f"NIST SP 800-82 path not found: {nist_pdf_path}")
        
    if not all_chunks:
        print("No chunks found to index.")
        return 0, 0
        
    print(f"Total chunks gathered: {len(all_chunks)}")
    
    # 2. Initialize ChromaDB and write vectors
    os.makedirs(db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Delete collection if it already exists to avoid duplicate accumulation
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing ChromaDB collection: {collection_name}")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name=collection_name)
    
    # Load model and embed
    model = get_embedding_model(model_name)
    
    texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    ids = [f"doc_{i}" for i in range(len(all_chunks))]
    
    print("Generating dense embeddings (this may take a few moments)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # Add to ChromaDB in batches to prevent payload size errors
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            documents=texts[i:end]
        )
    print("Successfully populated ChromaDB vector store.")
    
    # 3. Build and serialize BM25 index
    print("Building BM25 sparse index...")
    from rank_bm25 import BM25Okapi
    
    tokenized_corpus = [tokenize_text(text) for text in texts]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "bm25": bm25_model,
        "texts": texts,
        "metadatas": metadatas,
        "ids": ids
    }
    
    bm25_path = os.path.join(db_path, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"BM25 index successfully saved to {bm25_path}")
    
    return len(all_chunks), len(all_chunks)
