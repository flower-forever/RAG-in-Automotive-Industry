import os
import json
import re
import sys
from typing import List, Dict, Any

# Ensure project root is in sys.path to allow running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import SecureOpsRetriever
from src.generation import SecureOpsGenerator, sigmoid

import math


def precision_at_k(chunks: List[Dict[str, Any]], expected_source: str, k: int) -> float:
    if not chunks or expected_source == "NONE":
        return 0.0
    topk = chunks[:k]
    matches = sum(1 for c in topk if c.get("metadata", {}).get("source") == expected_source)
    return matches / float(k)


def mrr_at_k(chunks: List[Dict[str, Any]], expected_source: str, k: int) -> float:
    if not chunks or expected_source == "NONE":
        return 0.0
    for rank, c in enumerate(chunks[:k], start=1):
        if c.get("metadata", {}).get("source") == expected_source:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunks: List[Dict[str, Any]], expected_source: str, k: int) -> float:
    """Binary relevance nDCG@k for any number of relevant items."""
    if not chunks or expected_source == "NONE":
        return 0.0
    # Relevance vector for actual ranking
    rels = [1 if c.get("metadata", {}).get("source") == expected_source else 0 for c in chunks[:k]]
    dcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(rels, start=1))
    # Ideal DCG with all relevant documents at the top
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0

def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from strings."""
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

def run_evaluation(
    qa_path: str = "data/evaluation_qa.json",
    db_path: str = "chroma_db",
    collection_name: str = "secureops_assistant",
    report_output: str = "evaluation_report.md"
):
    print("====================================================")
    print("       SecureOps RAG Benchmarking & Evaluation      ")
    print("====================================================")
    
    if not os.path.exists(qa_path):
        print(f"Error: Evaluation QA dataset not found at {qa_path}")
        return
        
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    print(f"Loaded {len(qa_pairs)} evaluation cases.")
    
    # Check if database exists
    bm25_file = os.path.join(db_path, "bm25_index.pkl")
    if not os.path.exists(bm25_file):
        print(f"Error: Database index files not found at {db_path}. Please build the index first.")
        return
        
    print("Initializing retriever and generator...")
    retriever = SecureOpsRetriever(db_path=db_path, collection_name=collection_name)
    generator = SecureOpsGenerator()
    
    has_api_key = generator._has_client
    print(f"DeepSeek API Configured: {'Yes (Live generation evaluation enabled)' if has_api_key else 'No (Retrieval-only evaluation enabled - set DEEPSEEK_API_KEY to enable full eval)'}")
    
    results = []
    
    naive_hits = 0
    hybrid_hits = 0
    naive_rejections = 0
    hybrid_rejections = 0
    
    naive_coverage_scores = []
    hybrid_coverage_scores = []
    # New ranking metrics
    naive_precisions = []
    hybrid_precisions = []
    naive_mrrs = []
    hybrid_mrrs = []
    naive_ndcgs = []
    hybrid_ndcgs = []
    
    for qa in qa_pairs:
        qid = qa["id"]
        category = qa["category"]
        query = qa["question"]
        expected_source = qa["ground_truth_source"]
        keywords = qa["expected_keywords"]
        
        print(f"\n[{qid}/{len(qa_pairs)}] Category: {category} | Evaluating: '{query}'")
        
        # 1. Run Naive Retrieval (Pure Dense Search, top 10)
        naive_chunks = retriever.dense_search(query, top_k=10)
        
        # 2. Run Hybrid Retrieval (Dense + BM25 + RRF + Cross-Encoder Reranker, top 10)
        hybrid_chunks = retriever.retrieve(query, k=10)
        
        # Check retrieval hits (is expected source in retrieved sources?)
        naive_hit = False
        if expected_source != "NONE":
            naive_hit = any(chunk.get("metadata", {}).get("source") == expected_source for chunk in naive_chunks)
            if naive_hit:
                naive_hits += 1
                
        hybrid_hit = False
        if expected_source != "NONE":
            hybrid_hit = any(chunk.get("metadata", {}).get("source") == expected_source for chunk in hybrid_chunks)
            if hybrid_hit:
                hybrid_hits += 1

        # Compute ranking metrics for this query
        k = 10
        if expected_source != "NONE":
            naive_p = precision_at_k(naive_chunks, expected_source, k)
            hybrid_p = precision_at_k(hybrid_chunks, expected_source, k)
            naive_precisions.append(naive_p)
            hybrid_precisions.append(hybrid_p)

            naive_m = mrr_at_k(naive_chunks, expected_source, k)
            hybrid_m = mrr_at_k(hybrid_chunks, expected_source, k)
            naive_mrrs.append(naive_m)
            hybrid_mrrs.append(hybrid_m)

            naive_n = ndcg_at_k(naive_chunks, expected_source, k)
            hybrid_n = ndcg_at_k(hybrid_chunks, expected_source, k)
            naive_ndcgs.append(naive_n)
            hybrid_ndcgs.append(hybrid_n)
                
        # 3. Generation evaluation (if API Key is configured)
        naive_ans = ""
        hybrid_ans = ""
        naive_cov = 0.0
        hybrid_cov = 0.0
        naive_rejected = False
        hybrid_rejected = False
        
        # Rejection checking (even without API we can check confidence scores)
        # For naive search, we don't have rerank score, so we check distance (or mock threshold)
        # For hybrid, we check top rerank score
        top_rerank_score = hybrid_chunks[0].get("rerank_score", -99.0) if hybrid_chunks else -99.0
        hybrid_confidence = sigmoid(top_rerank_score) if hybrid_chunks else 0.0
        
        if category == "HONEST_REJECTION":
            if top_rerank_score < generator.low_confidence_threshold:
                hybrid_rejected = True
                hybrid_rejections += 1
            # Naive rejection based on distance (similarity < 0.4)
            top_naive_score = naive_chunks[0].get("score", 0.0) if naive_chunks else 0.0
            if top_naive_score < 0.4:
                naive_rejected = True
                naive_rejections += 1
                
        if has_api_key:
            # Generate Naive Answer
            naive_ans, _, _ = generator.generate_answer(query, naive_chunks)
            # Generate Hybrid Answer
            hybrid_ans, _, _ = generator.generate_answer(query, hybrid_chunks)
            
            # Strip ANSI
            naive_ans_clean = strip_ansi(naive_ans).lower()
            hybrid_ans_clean = strip_ansi(hybrid_ans).lower()
            
            # Check rejection string in answers
            rejection_phrase = "don't have enough information"
            if category == "HONEST_REJECTION":
                if rejection_phrase in naive_ans_clean:
                    naive_rejected = True
                if rejection_phrase in hybrid_ans_clean:
                    hybrid_rejected = True
            
            # Keyword coverage
            if category != "HONEST_REJECTION" and keywords:
                naive_matches = sum(1 for kw in keywords if kw.lower() in naive_ans_clean)
                naive_cov = naive_matches / len(keywords)
                naive_coverage_scores.append(naive_cov)
                
                hybrid_matches = sum(1 for kw in keywords if kw.lower() in hybrid_ans_clean)
                hybrid_cov = hybrid_matches / len(keywords)
                hybrid_coverage_scores.append(hybrid_cov)
        else:
            # Simulated coverage based on overlap of retrieved texts with expected keywords
            naive_text_all = " ".join([c["text"] for c in naive_chunks]).lower()
            hybrid_text_all = " ".join([c["text"] for c in hybrid_chunks]).lower()
            
            if category != "HONEST_REJECTION" and keywords:
                naive_matches = sum(1 for kw in keywords if kw.lower() in naive_text_all)
                naive_cov = naive_matches / len(keywords)
                naive_coverage_scores.append(naive_cov)
                
                hybrid_matches = sum(1 for kw in keywords if kw.lower() in hybrid_text_all)
                hybrid_cov = hybrid_matches / len(keywords)
                hybrid_coverage_scores.append(hybrid_cov)
                
        results.append({
            "id": qid,
            "category": category,
            "question": query,
            "expected_source": expected_source,
            "naive_hit": naive_hit if expected_source != "NONE" else "N/A",
            "hybrid_hit": hybrid_hit if expected_source != "NONE" else "N/A",
            "naive_cov": naive_cov,
            "hybrid_cov": hybrid_cov,
            "naive_rejected": naive_rejected if category == "HONEST_REJECTION" else "N/A",
            "hybrid_rejected": hybrid_rejected if category == "HONEST_REJECTION" else "N/A"
        })
        
    # Calculate summary metrics
    total_non_rejection = sum(1 for qa in qa_pairs if qa["category"] != "HONEST_REJECTION")
    total_rejection = len(qa_pairs) - total_non_rejection
    
    naive_hit_rate = (naive_hits / total_non_rejection) * 100
    hybrid_hit_rate = (hybrid_hits / total_non_rejection) * 100
    
    naive_rejection_acc = (naive_rejections / total_rejection) * 100 if total_rejection > 0 else 100
    hybrid_rejection_acc = (hybrid_rejections / total_rejection) * 100 if total_rejection > 0 else 100
    
    avg_naive_cov = (sum(naive_coverage_scores) / len(naive_coverage_scores)) * 100 if naive_coverage_scores else 0.0
    avg_hybrid_cov = (sum(hybrid_coverage_scores) / len(hybrid_coverage_scores)) * 100 if hybrid_coverage_scores else 0.0

    avg_naive_prec = (sum(naive_precisions) / len(naive_precisions)) * 100 if naive_precisions else 0.0
    avg_hybrid_prec = (sum(hybrid_precisions) / len(hybrid_precisions)) * 100 if hybrid_precisions else 0.0

    avg_naive_mrr = (sum(naive_mrrs) / len(naive_mrrs)) if naive_mrrs else 0.0
    avg_hybrid_mrr = (sum(hybrid_mrrs) / len(hybrid_mrrs)) if hybrid_mrrs else 0.0

    avg_naive_ndcg = (sum(naive_ndcgs) / len(naive_ndcgs)) if naive_ndcgs else 0.0
    avg_hybrid_ndcg = (sum(hybrid_ndcgs) / len(hybrid_ndcgs)) if hybrid_ndcgs else 0.0
    
    # Output markdown report
    report = []
    report.append("# SecureOps Assistant — RAG Benchmarking Report")
    report.append(f"\nThis report compares the performance of **Naive Vector Search** against **Advanced Hybrid Search** (Dense + Sparse + RRF + Reranking) across {len(qa_pairs)} evaluation cases.")
    
    report.append("\n## 📊 Summary Metrics")
    report.append("\n| Metric | Naive Vector Search (Baseline) | Advanced Hybrid Search (Modular RAG) | Improvement |")
    report.append("| :--- | :---: | :---: | :---: |")
    report.append(f"| **Retrieval Hit Rate** | {naive_hit_rate:.1f}% | {hybrid_hit_rate:.1f}% | +{hybrid_hit_rate - naive_hit_rate:.1f}% |")
    report.append(f"| **Keyword Factual Coverage** | {avg_naive_cov:.1f}% | {avg_hybrid_cov:.1f}% | +{avg_hybrid_cov - avg_naive_cov:.1f}% |")
    report.append(f"| **Honest Rejection Accuracy** | {naive_rejection_acc:.1f}% | {hybrid_rejection_acc:.1f}% | +{hybrid_rejection_acc - naive_rejection_acc:.1f}% |")
    report.append(f"| **Precision@10** | {avg_naive_prec:.1f}% | {avg_hybrid_prec:.1f}% | +{avg_hybrid_prec - avg_naive_prec:.1f}% |")
    report.append(f"| **MRR@10** | {avg_naive_mrr:.3f} | {avg_hybrid_mrr:.3f} | +{avg_hybrid_mrr - avg_naive_mrr:.3f} |")
    report.append(f"| **nDCG@10** | {avg_naive_ndcg:.3f} | {avg_hybrid_ndcg:.3f} | +{avg_hybrid_ndcg - avg_naive_ndcg:.3f} |")
    
    report.append("\n## 📝 Detailed Evaluation Results")
    report.append("\n| ID | Category | Question | Expected Source | Naive Hit? | Hybrid Hit? | Naive Cover | Hybrid Cover | Rejection (N/H)? |")
    report.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for r in results:
        naive_hit_str = "✅" if r["naive_hit"] is True else ("❌" if r["naive_hit"] is False else "—")
        hybrid_hit_str = "✅" if r["hybrid_hit"] is True else ("❌" if r["hybrid_hit"] is False else "—")
        
        rej_str = "—"
        if r["category"] == "HONEST_REJECTION":
            n_rej = "✅" if r["naive_rejected"] else "❌"
            h_rej = "✅" if r["hybrid_rejected"] else "❌"
            rej_str = f"{n_rej} / {h_rej}"
            
        report.append(
            f"| {r['id']} | {r['category']} | {r['question']} | {r['expected_source']} | "
            f"{naive_hit_str} | {hybrid_hit_str} | {r['naive_cov']*100:.0f}% | {r['hybrid_cov']*100:.0f}% | {rej_str} |"
        )
        
    report_text = "\n".join(report)
    
    # Save report
    with open(report_output, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\nEvaluation complete. Benchmark report saved to: {report_output}")
    print("\n================== BENCHMARK SUMMARY ==================")
    print(f"Retrieval Hit Rate:     Naive: {naive_hit_rate:.1f}%  |  Hybrid: {hybrid_hit_rate:.1f}%")
    print(f"Keyword Coverage:       Naive: {avg_naive_cov:.1f}%  |  Hybrid: {avg_hybrid_cov:.1f}%")
    print(f"Honest Rejection Acc:   Naive: {naive_rejection_acc:.1f}%  |  Hybrid: {hybrid_rejection_acc:.1f}%")
    print(f"Precision@10:            Naive: {avg_naive_prec:.1f}%  |  Hybrid: {avg_hybrid_prec:.1f}%")
    print(f"MRR@10:                  Naive: {avg_naive_mrr:.3f}  |  Hybrid: {avg_hybrid_mrr:.3f}")
    print(f"nDCG@10:                 Naive: {avg_naive_ndcg:.3f}  |  Hybrid: {avg_hybrid_ndcg:.3f}")
    print("=======================================================")

if __name__ == "__main__":
    run_evaluation()
