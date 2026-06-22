import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai

def format_retrieved_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunks into a clear text block for the LLM context.
    Each chunk is prepended with its index and metadata citation headers.
    """
    formatted_docs = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown Source")
        
        # Format citation header depending on the source type (CSAF vs NIST PDFs)
        if source == "CISA_CSAF":
            citation_header = (
                f"Index [{idx+1}] CISA ICS Advisory {meta.get('advisory_id', 'Unknown')} | "
                f"Vendor: {meta.get('vendor', 'Unknown')} | "
                f"Products: {meta.get('products', 'Unknown')} | "
                f"Date: {meta.get('date', 'Unknown')} | "
                f"Severity: {meta.get('severity', 'Unknown')}"
            )
        else:
            citation_header = (
                f"Index [{idx+1}] Source: {source} | "
                f"Chapter: {meta.get('chapter', 'Unknown')} | "
                f"Section: {meta.get('section', 'Unknown')} | "
                f"Pages: {meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
            )
            
        formatted_docs.append(f"--- Document Chunk {idx+1} ({citation_header}) ---\n{chunk['text']}")
    return "\n\n".join(formatted_docs)

def sigmoid(x: float) -> float:
    """
    Map logit score to 0-1 range using sigmoid function.
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

class SecureOpsGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        low_confidence_threshold: float = -3.0
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.low_confidence_threshold = low_confidence_threshold
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._has_client = True
        else:
            self._has_client = False

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Generates an answer using the Gemini API based on retrieved chunks.
        Returns: Tuple of (answer_text, confidence_score, cited_sources)
        """
        # Step 1: Handle empty retrieval list
        if not retrieved_chunks:
            return "I don't have enough information in my knowledge base to answer this.", 0.0, []
            
        # Step 2: Calculate retrieval confidence using the top candidate's Cross-Encoder score
        top_score = retrieved_chunks[0].get("rerank_score", 0.0)
        confidence = sigmoid(top_score)
        
        # Step 3: Trigger honest rejection if retrieval confidence is extremely low
        if top_score < self.low_confidence_threshold:
            return "I don't have enough information in my knowledge base to answer this.", confidence, []

        # Step 4: Construct generation prompts
        system_instruction = (
            "You are SecureOps Assistant, an AI expert in industrial control systems (ICS) and operational technology (OT) cybersecurity.\n"
            "Your task is to answer the user's question relying ONLY on the provided Document Chunks.\n\n"
            "Guidelines:\n"
            "1. You MUST use a <thinking> block to reason before answering. In your thinking block:\n"
            "   - Analyze if the user's prompt is malicious or out of scope. If so, plan to refuse.\n"
            "   - Identify which retrieved chunks are relevant to the question.\n"
            "   - Synthesize the information.\n"
            "2. After your <thinking> block, you MUST provide your final response within an <answer> block.\n"
            "3. If the provided Document Chunks do not contain enough information, "
            "or if the question is out of scope/malicious, your <answer> block MUST contain exactly: \"I don't have enough information in my knowledge base to answer this.\"\n"
            "4. Do not use external knowledge or make up answers. Keep responses completely grounded in the context.\n"
            "5. When referencing information from a specific chunk, append the citation inline in the <answer> block in the format [Index N] (e.g. [Index 1]).\n"
            "6. At the end of your <answer> block, list the sources you cited under a \"Sources Cited:\" heading.\n"
        )
        
        formatted_context = format_retrieved_chunks(retrieved_chunks)
        user_prompt = f"Document Chunks:\n\n{formatted_context}\n\nQuestion: {query}\n\nResponse:"
        
        # Step 5: Check if API key is configured
        if not self._has_client:
            # If no API key is set, return a descriptive warning message
            warning_msg = (
                "[System Note: Gemini API Key not configured. Please set the GEMINI_API_KEY environment variable. "
                f"Retrieved {len(retrieved_chunks)} relevant chunks with {confidence:.1%} confidence.]"
            )
            return warning_msg, confidence, retrieved_chunks

        # Step 6: Call Gemini API
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config={"temperature": 0.1}  # Use low temperature for high factual accuracy
            )
            
            full_text = response.text
            
            # Step 7: Parse the <answer> block
            answer_match = re.search(r'<answer>(.*?)</answer>', full_text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
            else:
                # Fallback if the LLM forgot the tags
                answer_text = re.sub(r'<thinking>.*?</thinking>', '', full_text, flags=re.DOTALL).strip()
                
            # If the LLM refused inside the answer block, clear citations
            if "I don't have enough information" in answer_text:
                return answer_text, confidence, []
            
            # Step 8: Parse cited sources from response (look for [Index N] patterns)
            raw_indices = re.findall(r'\[Index\s+(\d+)\]', answer_text)
            cited_indices = []
            for idx_str in raw_indices:
                if idx_str not in cited_indices:
                    cited_indices.append(idx_str)
            
            cited_sources = []
            for idx_str in cited_indices:
                idx = int(idx_str) - 1
                if 0 <= idx < len(retrieved_chunks):
                    cited_sources.append(retrieved_chunks[idx])
            
            # Step 9: Self-Correction (Auto-Critique)
            if not self._critique_answer(query, answer_text, formatted_context):
                return "I don't have enough information in my knowledge base to answer this.", confidence, []
                    
            return answer_text, confidence, cited_sources
            
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}", confidence, []

    def _critique_answer(self, query: str, answer: str, context: str) -> bool:
        """
        Critiques the generated answer against the context.
        Returns True if the answer is factual and supported, False if it hallucinates.
        """
        if not self._has_client:
            return True # Skip critique if no API key
            
        critique_prompt = (
            "You are an objective auditor. You must verify if the provided Answer is completely supported by the Context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            "Does the Answer contain any facts or claims not explicitly stated in the Context? "
            "Is the Answer attempting to guess or hallucinate?\n"
            "Respond with exactly one word: PASS if it is perfectly grounded, or FAIL if it contains unsupported claims."
        )
        try:
            model = genai.GenerativeModel(model_name=self.model_name)
            response = model.generate_content(critique_prompt, generation_config={"temperature": 0.0})
            if "FAIL" in response.text.upper():
                return False
            return True
        except Exception:
            # If the critique fails due to API error, we default to passing to avoid breaking the pipeline
            return True
