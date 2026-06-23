from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import os
from src.retrieval import SecureOpsRetriever
from src.generation import SecureOpsGenerator
from src.indexing import build_index
from src.query_rewrite import rewrite_query

app = FastAPI(title="SecureOps RAG API")

# Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Global instances
retriever = None
generator = None

class AskRequest(BaseModel):
    query: str
    vendor: Optional[str] = None
    severity: Optional[str] = None
    source: Optional[str] = None

def get_models():
    global retriever, generator
    if retriever is None or generator is None:
        retriever = SecureOpsRetriever()
        generator = SecureOpsGenerator()
    return retriever, generator

@app.post("/api/ask")
def ask_question(req: AskRequest):
    r, g = get_models()
    candidates = rewrite_query(req.query)
    queries_list = [req.query] + [c["text"] for c in candidates if c["text"].lower() != req.query.lower()]
    
    retrieved = r.retrieve(queries_list, k=5, vendor=req.vendor, severity=req.severity, source=req.source)
    answer, confidence, cited = g.generate_answer(req.query, retrieved)
    
    return {
        "answer": answer,
        "confidence": confidence,
        "cited": cited,
        "expanded_queries": queries_list
    }

from fastapi.responses import StreamingResponse

@app.post("/api/ask_stream")
def ask_question_stream(req: AskRequest):
    r, g = get_models()
    candidates = rewrite_query(req.query)
    queries_list = [req.query] + [c["text"] for c in candidates if c["text"].lower() != req.query.lower()]
    
    retrieved = r.retrieve(queries_list, k=5, vendor=req.vendor, severity=req.severity, source=req.source)
    
    # We yield the expanded queries as the very first chunk in a special metadata format 
    # so the frontend knows what was queried.
    def stream_generator():
        import json
        yield f"__EXPANDED_QUERIES__:{json.dumps(queries_list)}\n\n"
        for chunk in g.generate_answer_stream(req.query, retrieved):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/api/upload")
def upload_files(files: List[UploadFile] = File(...)):
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Max 3 files allowed.")
    
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    for f in files:
        content = f.file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File {f.filename} exceeds 5MB limit.")
        
        file_path = os.path.join(upload_dir, f.filename)
        with open(file_path, "wb") as out:
            out.write(content)
            
    global retriever, generator
    # For hackathon purpose we only index the provided core directories, 
    # but ideally we would also index the upload_dir.
    build_index(csaf_dir="doc/cisa_csaf", csf_pdf_path="doc/NIST Cybersecurity Framework(CSF) 2.0.pdf", nist_pdf_path="doc/NIST.SP.800-82r3.pdf", limit_pdf_pages=True)
    
    retriever = None
    generator = None
    
    return {"status": "success", "message": f"{len(files)} files uploaded and index rebuilt."}
