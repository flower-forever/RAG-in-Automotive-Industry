import cmd
import os
import sys
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path to allow running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indexing import build_index
from src.retrieval import SecureOpsRetriever
from src.generation import SecureOpsGenerator
from src.query_rewrite import rewrite_query

# ANSI Colors
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[1;31m"
COLOR_GREEN = "\033[1;32m"
COLOR_YELLOW = "\033[1;33m"
COLOR_BLUE = "\033[1;34m"
COLOR_CYAN = "\033[1;36m"
COLOR_WHITE = "\033[1;37m"
COLOR_GRAY = "\033[90m"

ASCII_ART = r"""  ____                               ___               
 / ___|  ___  ___ _   _ _ __ ___    / _ \ _ __  ___   
 \___ \ / _ \/ __| | | | '__/ _ \  | | | | '_ \/ __|  
  ___) |  __/ (__| |_| | | |  __/  | |_| | |_) \__ \ 
 |____/ \___|\___|\__,_|_|  \___|   \___/| .__/|___/  
                                         |_|          """

ASCII_BANNER = f"{COLOR_CYAN}===================================================================\n" + ASCII_ART + f"\n           SecureOps RAG Assistant (Industrial Cybersecurity)\n==================================================================={COLOR_RESET}"

class SecureOpsCLI(cmd.Cmd):
    intro = ASCII_BANNER + f"\nType {COLOR_GREEN}help{COLOR_RESET} to list commands. Models load lazily on first query.\n"
    
    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: str = "secureops_assistant",
        csaf_dir: str = "doc/cisa_csaf",
        csf_pdf_path: str = "doc/NIST Cybersecurity Framework(CSF) 2.0.pdf",
        nist_pdf_path: str = "doc/NIST.SP.800-82r3.pdf"
    ):
        super().__init__()
        self.db_path = db_path
        self.collection_name = collection_name
        self.csaf_dir = csaf_dir
        self.csf_pdf_path = csf_pdf_path
        self.nist_pdf_path = nist_pdf_path
        
        # State variables
        self.filters: Dict[str, Optional[str]] = {
            "vendor": None,
            "severity": None,
            "source": None
        }
        self.query_history: List[str] = []
        
        # Lazy loaded components
        self.retriever: Optional[SecureOpsRetriever] = None
        self.generator: Optional[SecureOpsGenerator] = None
        
        self.update_prompt()
        self.check_database_status()

    def check_database_status(self):
        """Check if indices exist and display status message."""
        bm25_file = os.path.join(self.db_path, "bm25_index.pkl")
        if os.path.exists(bm25_file):
            print(f"[{COLOR_GREEN}System{COLOR_RESET}]: Index files found. Ready for queries.")
        else:
            print(f"[{COLOR_YELLOW}Warning{COLOR_RESET}]: No index database found. Please run the {COLOR_CYAN}rebuild{COLOR_RESET} command first.")

    def update_prompt(self):
        """Update prompt string dynamically to reflect active filters."""
        active_filters = []
        for key, val in self.filters.items():
            if val:
                active_filters.append(f"{key}:{val}")
        
        if active_filters:
            filter_str = f" [{COLOR_YELLOW}{', '.join(active_filters)}{COLOR_CYAN}]"
        else:
            filter_str = ""
            
        self.prompt = f"{COLOR_CYAN}SecureOps{filter_str}>{COLOR_RESET} "

    def emptyline(self):
        """Do nothing on empty line entry."""
        pass

    def ensure_models_loaded(self) -> bool:
        """Lazily load SentenceTransformer, Cross-Encoder, and Gemini generator."""
        if self.retriever is not None and self.generator is not None:
            return True
            
        print(f"[{COLOR_CYAN}System{COLOR_RESET}]: Loading embedding and reranker models. This may take a few seconds...")
        try:
            self.retriever = SecureOpsRetriever(
                db_path=self.db_path,
                collection_name=self.collection_name
            )
            # Generator automatically picks up GEMINI_API_KEY from environment or constructor
            self.generator = SecureOpsGenerator()
            print(f"[{COLOR_GREEN}System{COLOR_RESET}]: Models loaded successfully.")
            return True
        except FileNotFoundError as e:
            print(f"{COLOR_RED}Error loading index database:{COLOR_RESET} {str(e)}")
            print(f"Please run the {COLOR_CYAN}rebuild{COLOR_RESET} command to create index files first.")
            return False
        except Exception as e:
            print(f"{COLOR_RED}Unexpected error loading models:{COLOR_RESET} {str(e)}")
            return False

    def do_ask(self, arg: str):
        """Ask a cybersecurity query to the RAG assistant.
        Usage: ask <your question here>
        """
        if not arg.strip():
            print(f"[{COLOR_RED}Error{COLOR_RESET}]: Please provide a question. E.g., 'ask What is PR.AC?'")
            return
            
        if not self.ensure_models_loaded():
            return
            
        query = arg.strip()
        self.query_history.append(query)
        
        print(f"\n{COLOR_CYAN}Retrieving context chunks...{COLOR_RESET}")
        try:
            # 1. Generate rewrite candidates and perform retrieval for each candidate.
            #    We will pick the candidate with the highest top rerank_score (or score) as best.
            candidates = rewrite_query(query)
            # Ensure the original query is among candidates (as fallback)
            orig_present = any(c["text"].strip().lower() == query.strip().lower() for c in candidates)
            if not orig_present:
                candidates.insert(0, {"type": "original", "text": query})

            candidate_results = []
            with ThreadPoolExecutor(max_workers=min(6, len(candidates))) as executor:
                future_to_candidate = {}
                for c in candidates:
                    qtext = c["text"]
                    future = executor.submit(
                        self.retriever.retrieve,
                        query=qtext,
                        k=5,
                        vendor=self.filters["vendor"],
                        severity=self.filters["severity"],
                        source=self.filters["source"]
                    )
                    future_to_candidate[future] = (c, qtext)

                for future in as_completed(future_to_candidate):
                    c, qtext = future_to_candidate[future]
                    try:
                        res = future.result()
                    except Exception:
                        res = []
                    top_score = 0.0
                    if res:
                        top_score = float(res[0].get("rerank_score", res[0].get("score", 0.0)))
                    candidate_results.append({"candidate": c, "query": qtext, "retrieved": res, "top_score": top_score})

            # Choose best candidate by top_score (reranker score) — fallback to original if none better.
            best = max(candidate_results, key=lambda x: x["top_score"]) if candidate_results else None
            if best is None:
                retrieved = []
            else:
                retrieved = best["retrieved"]

            # 2. Generate answer using best candidate's retrieval
            print(f"{COLOR_CYAN}Generating response via LLM...{COLOR_RESET}\n")
            answer, confidence, cited = self.generator.generate_answer(
                query=(best["query"] if best else query),
                retrieved_chunks=retrieved
            )
            
            # 3. Print answer
            print(f"{COLOR_BOLD}{COLOR_WHITE}Question:{COLOR_RESET} {query}")
            print(f"{COLOR_BOLD}{COLOR_WHITE}Answer:{COLOR_RESET}\n{answer}\n")
            
            # 4. Print confidence metrics
            conf_percent = confidence * 100
            if confidence >= 0.7:
                conf_color = COLOR_GREEN
            elif confidence >= 0.3:
                conf_color = COLOR_YELLOW
            else:
                conf_color = COLOR_RED
            print(f"{COLOR_BOLD}{COLOR_WHITE}Retrieval Confidence:{COLOR_RESET} {conf_color}{conf_percent:.1f}%{COLOR_RESET}")
            
            # 5. Print cited sources details
            if cited:
                print(f"\n{COLOR_BOLD}{COLOR_WHITE}Sources Details:{COLOR_RESET}")
                for idx, chunk in enumerate(cited):
                    meta = chunk.get("metadata", {})
                    source = meta.get("source", "Unknown")
                    print(f" {COLOR_GRAY}- [{idx+1}] Source: {source}{COLOR_RESET}")
                    if source == "CISA_CSAF":
                        print(f"   Advisory ID: {meta.get('advisory_id', 'N/A')} | Vendor: {meta.get('vendor', 'N/A')} | Severity: {meta.get('severity', 'N/A')}")
                    else:
                        print(f"   Chapter: {meta.get('chapter', 'N/A')} | Section: {meta.get('section', 'N/A')} | Page(s): {meta.get('page_start', '?')}-{meta.get('page_end', '?')}")
            else:
                print(f"\n{COLOR_YELLOW}No sources were explicitly cited for this answer.{COLOR_RESET}")
            # Print which rewrite candidate was used (if available)
            try:
                if best and best.get("candidate"):
                    c = best["candidate"]
                    print(f"\n{COLOR_BOLD}{COLOR_WHITE}Rewrite Used:{COLOR_RESET} ({c['type']}) {best['query']}")
                    # also print top scores of other candidates for debugging
                    print(f"{COLOR_BOLD}{COLOR_WHITE}Candidate Scores:{COLOR_RESET}")
                    for cr in candidate_results:
                        t = cr["candidate"]["type"]
                        q = cr["query"]
                        s = cr["top_score"]
                        print(f"  - {t}: score={s:.4f} | {q}")
            except Exception:
                pass
            print()
            
        except Exception as e:
            print(f"{COLOR_RED}Error running query:{COLOR_RESET} {str(e)}")

    def do_rewrite(self, arg: str):
        """Rewrite and display alternate formulations of a question.
        Usage: rewrite <your question here>
        """
        query = arg.strip()
        if not query:
            print(f"{COLOR_RED}Error{COLOR_RESET}: Please provide a question to rewrite. E.g., 'rewrite What is PR.AC?'")
            return

        try:
            candidates = rewrite_query(query)
            print(f"\nOriginal: {query}\n")
            print("Rewritten candidates:")
            for i, c in enumerate(candidates, start=1):
                print(f" [{i}] ({c['type']}) {c['text']}")
            print()
        except Exception as e:
            print(f"{COLOR_RED}Error generating rewrites:{COLOR_RESET} {e}")

    def do_filter(self, arg: str):
        """Manage active metadata filters.
        Usage:
          filter                          - View active filters
          filter vendor <vendor_name>     - Filter CSAF advisories by vendor
          filter severity <HIGH|MEDIUM>   - Filter CSAF advisories by severity
          filter source <source_name>     - Filter by source (CISA_CSAF, NIST_CSF_2.0, NIST_SP_800-82_R3)
          filter clear                    - Clear all filters
        """
        parts = arg.strip().split(maxsplit=1)
        if not parts:
            # Show current filters
            print(f"\n{COLOR_BOLD}{COLOR_WHITE}Active Filters:{COLOR_RESET}")
            active = False
            for k, v in self.filters.items():
                if v:
                    print(f"  {COLOR_CYAN}{k}{COLOR_RESET}: {v}")
                    active = True
            if not active:
                print("  No active filters.")
            print()
            return
            
        cmd_type = parts[0].lower()
        if cmd_type == "clear":
            self.filters = {"vendor": None, "severity": None, "source": None}
            print(f"[{COLOR_GREEN}System{COLOR_RESET}]: All filters cleared.")
            self.update_prompt()
            return
            
        if len(parts) < 2:
            print(f"[{COLOR_RED}Error{COLOR_RESET}]: Missing filter value. E.g. 'filter vendor Siemens'")
            return
            
        value = parts[1].strip()
        if cmd_type in self.filters:
            self.filters[cmd_type] = value
            print(f"[{COLOR_GREEN}System{COLOR_RESET}]: Filter for {COLOR_CYAN}{cmd_type}{COLOR_RESET} set to '{value}'.")
            self.update_prompt()
        else:
            print(f"[{COLOR_RED}Error{COLOR_RESET}]: Unknown filter type '{cmd_type}'. Supported: vendor, severity, source.")

    def do_history(self, arg: str):
        """Show previous queries asked in the current session.
        Usage: history
        """
        if not self.query_history:
            print("No queries in history yet.")
            return
            
        print(f"\n{COLOR_BOLD}{COLOR_WHITE}Query History:{COLOR_RESET}")
        for idx, q in enumerate(self.query_history):
            print(f"  {idx+1}. {q}")
        print()

    def do_sources(self, arg: str):
        """Display statistic summaries of the indexed documents from BM25 index file.
        Usage: sources
        """
        bm25_file = os.path.join(self.db_path, "bm25_index.pkl")
        if not os.path.exists(bm25_file):
            print(f"[{COLOR_RED}Error{COLOR_RESET}]: No index file found. Please run the {COLOR_CYAN}rebuild{COLOR_RESET} command first.")
            return
            
        try:
            with open(bm25_file, "rb") as f:
                bm25_data = pickle.load(f)
                metadatas = bm25_data.get("metadatas", [])
                
            total_chunks = len(metadatas)
            counts: Dict[str, int] = {}
            vendors: Dict[str, int] = {}
            
            for meta in metadatas:
                src = meta.get("source", "Unknown")
                counts[src] = counts.get(src, 0) + 1
                if src == "CISA_CSAF":
                    vendor = meta.get("vendor", "Unknown")
                    vendors[vendor] = vendors.get(vendor, 0) + 1
                    
            print(f"\n{COLOR_BOLD}{COLOR_WHITE}Index Data Summary:{COLOR_RESET}")
            print(f"  Total Chunks Indexed: {total_chunks}")
            print(f"\n{COLOR_BOLD}{COLOR_WHITE}Sources Breakdown:{COLOR_RESET}")
            for src, count in counts.items():
                print(f"  - {COLOR_CYAN}{src}{COLOR_RESET}: {count} chunks")
                
            if vendors:
                print(f"\n{COLOR_BOLD}{COLOR_WHITE}Top CSAF Vendors:{COLOR_RESET}")
                sorted_vendors = sorted(vendors.items(), key=lambda x: x[1], reverse=True)
                for vendor, count in sorted_vendors[:10]:
                    print(f"  - {COLOR_CYAN}{vendor}{COLOR_RESET}: {count} chunks")
            print()
            
        except Exception as e:
            print(f"{COLOR_RED}Error loading sources summary:{COLOR_RESET} {str(e)}")

    def do_rebuild(self, arg: str):
        """Parse all document directories and rebuild ChromaDB & BM25 indexes.
        Usage: rebuild [quick]
        Note: Passing 'quick' parses only a small subset of pages from NIST PDFs for speed.
        """
        quick_mode = (arg.strip().lower() == "quick")
        mode_desc = f"{COLOR_YELLOW}Quick Mode{COLOR_RESET} (sample subset of PDF pages)" if quick_mode else "Full Mode"
        
        confirm = input(f"Are you sure you want to rebuild the index database in {mode_desc}? (y/N): ")
        if confirm.lower() not in ('y', 'yes'):
            print("Rebuild canceled.")
            return
            
        print(f"\n{COLOR_CYAN}Starting database rebuild...{COLOR_RESET}")
        try:
            # Build index
            num_docs, num_chunks = build_index(
                csaf_dir=self.csaf_dir,
                csf_pdf_path=self.csf_pdf_path,
                nist_pdf_path=self.nist_pdf_path,
                db_path=self.db_path,
                collection_name=self.collection_name,
                limit_pdf_pages=quick_mode
            )
            
            # Reset retriever and generator so they reload indices on next ask
            self.retriever = None
            self.generator = None
            
            print(f"\n[{COLOR_GREEN}Success{COLOR_RESET}]: Database rebuilt successfully.")
            print(f"Indexed {num_chunks} chunks.")
            self.check_database_status()
            
        except Exception as e:
            print(f"{COLOR_RED}Rebuild failed:{COLOR_RESET} {str(e)}")

    def do_exit(self, arg: str) -> bool:
        """Exit the SecureOps Assistant CLI interface."""
        print("Goodbye!")
        return True

    def do_quit(self, arg: str) -> bool:
        """Exit the SecureOps Assistant CLI interface."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Exit on Ctrl-D."""
        print()
        return self.do_exit(arg)

if __name__ == "__main__":
    cli = SecureOpsCLI()
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
