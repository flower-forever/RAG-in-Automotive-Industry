import os
import re
import pickle
from typing import List, Dict, Any, Optional, Union
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.indexing import tokenize_text, get_torch_device

class SecureOpsRetriever:
    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: str = "secureops_assistant",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # 1. Initialize ChromaDB client
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database path not found: {db_path}. Please run indexing first.")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        # 2. Load BM25 index
        self.bm25_path = os.path.join(db_path, "bm25_index.pkl")
        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError(f"BM25 index file not found at {self.bm25_path}. Please run indexing first.")
        
        with open(self.bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.texts = bm25_data["texts"]
            self.metadatas = bm25_data["metadatas"]
            self.ids = bm25_data["ids"]
            
        # Map id to index for quick lookup
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.ids)}
        
        # 3. Load embedding model and cross-encoder reranker with device fallback
        device = get_torch_device()
        print(f"Loading retriever models on {device}...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        except Exception as exc:
            if device != "cpu":
                print(f"[Warning] Failed to load embedding model on {device}; falling back to CPU. Error: {exc}")
                self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
            else:
                raise

        try:
            self.reranker = CrossEncoder(reranker_model_name, device=device)
        except Exception as exc:
            if device != "cpu":
                print(f"[Warning] Failed to load reranker on {device}; falling back to CPU. Error: {exc}")
                self.reranker = CrossEncoder(reranker_model_name, device="cpu")
            else:
                raise

    def _match_metadata_filter(self, metadata: Dict[str, Any], vendor: Optional[str] = None, severity: Optional[str] = None, source: Optional[str] = None) -> bool:
        """
        Helper function to evaluate metadata filters for BM25 results.
        """
        if vendor:
            # Case-insensitive substring match to handle variations like 'Danelec' vs 'Danelec A/S'
            if vendor.lower() not in metadata.get("vendor", "").lower():
                return False
        if severity:
            if severity.upper() != metadata.get("severity", "").upper():
                return False
        if source:
            if source.lower() != metadata.get("source", "").lower():
                return False
        return True

    def dense_search(self, query: str, top_k: int = 20, vendor: Optional[str] = None, severity: Optional[str] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform dense vector search in ChromaDB.
        """
        # Apply query instruction for BGE model
        query_instruction = "Represent this sentence for searching relevant passages: "
        query_emb = self.embedding_model.encode(query_instruction + query, normalize_embeddings=True)
        
        # Construct ChromaDB metadata filter
        where_filters = []
        if vendor:
            where_filters.append({"vendor": vendor})
        if severity:
            where_filters.append({"severity": severity.upper()})
        if source:
            where_filters.append({"source": source})
            
        where_clause = None
        if len(where_filters) == 1:
            where_clause = where_filters[0]
        elif len(where_filters) > 1:
            where_clause = {"$and": where_filters}
            
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        dense_hits = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                text = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to similarity score
                similarity = 1.0 - dist
                dense_hits.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "score": similarity
                })
        return dense_hits

    def sparse_search(self, query: str, top_k: int = 20, vendor: Optional[str] = None, severity: Optional[str] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform BM25 search and apply metadata filters.
        """
        query_tokens = tokenize_text(query)
        scores = self.bm25.get_scores(query_tokens)
        
        sparse_hits = []
        for idx, score in enumerate(scores):
            # Skip documents with score of 0 (no matching tokens) for speed
            if score <= 0.0:
                continue
            meta = self.metadatas[idx]
            if self._match_metadata_filter(meta, vendor, severity, source):
                sparse_hits.append({
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": meta,
                    "score": score
                })
                
        # Sort by BM25 score descending
        sparse_hits.sort(key=lambda x: x["score"], reverse=True)
        return sparse_hits[:top_k]

    def reciprocal_rank_fusion(self, *hit_lists: List[Dict[str, Any]], rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).
        """
        rrf_scores = {}
        doc_details = {}
        
        for hits in hit_lists:
            for rank, hit in enumerate(hits):
                doc_id = hit["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
                doc_details[doc_id] = (hit["text"], hit["metadata"])
            
        # Compile and sort fused results
        fused_hits = []
        for doc_id, rrf_score in rrf_scores.items():
            text, meta = doc_details[doc_id]
            fused_hits.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "rrf_score": rrf_score
            })
            
        fused_hits.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_hits

    def retrieve(
        self,
        query_or_queries: Union[str, List[str]],
        k: int = 5,
        dense_ratio: float = 0.7,
        vendor: Optional[str] = None,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        top_n_candidates: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Modular retrieval entry point:
        1. Dense retrieval (ChromaDB)
        2. Sparse retrieval (BM25)
        3. Reciprocal Rank Fusion (RRF)
        4. Cross-Encoder Reranking
        Returns top k chunks.
        """
        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries

        # Step 1: Perform Dense and Sparse retrieval for all queries
        all_hit_lists = []
        for q in queries:
            dense_hits = self.dense_search(q, top_k=top_n_candidates, vendor=vendor, severity=severity, source=source)
            sparse_hits = self.sparse_search(q, top_k=top_n_candidates, vendor=vendor, severity=severity, source=source)
            all_hit_lists.extend([dense_hits, sparse_hits])
        
        # Step 2: reciprocal rank fusion (RRF)
        # We set rrf_k to 60 as recommended in literature
        fused_hits = self.reciprocal_rank_fusion(*all_hit_lists, rrf_k=60)
        
        # If no hits found at all, return empty
        if not fused_hits:
            return []
            
        # Step 3: Reranking with Cross-Encoder
        # Limit candidates for reranking to improve latency
        candidates = fused_hits[:top_n_candidates]
        
        primary_query = queries[0]
        candidate_pairs = [(primary_query, hit["text"]) for hit in candidates]
        rerank_scores = self.reranker.predict(candidate_pairs)
        
        # Attach cross-encoder scores and sort
        for idx, hit in enumerate(candidates):
            hit["rerank_score"] = float(rerank_scores[idx])
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top k
        return candidates[:k]
