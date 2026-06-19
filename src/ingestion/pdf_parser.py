import os
import re
import pymupdf4llm
from typing import List, Dict, Any

def clean_title(title_text: str) -> str:
    """
    Remove markdown formatting like asterisks and underscores, and strip whitespace.
    """
    cleaned = re.sub(r'[\*_`]', '', title_text)
    return cleaned.strip()

def update_header_path(header_path: List[str], level: int, title: str) -> List[str]:
    """
    Update the active hierarchical header path when a new heading is encountered.
    Truncates path to the parent level and appends the new title.
    """
    # Level is 1-indexed (e.g. # is level 1, ## is level 2)
    new_path = header_path[:level - 1]
    while len(new_path) < level - 1:
        new_path.append("Unknown Section")
    new_path.append(title)
    return new_path

def create_chunk(text: str, source_name: str, header_path: List[str], page_start: int, page_end: int) -> Dict[str, Any]:
    """
    Formulate a RAG chunk with metadata and a prepended hierarchical context block.
    """
    headers = [h for h in header_path if h]
    if not headers:
        headers = ["Introduction / Preface"]
        
    chapter = headers[0]
    section = headers[1] if len(headers) > 1 else ""
    subsection = headers[2] if len(headers) > 2 else ""
    
    metadata = {
        "source": source_name,
        "chapter": chapter,
        "section": section,
        "subsection": subsection,
        "page_start": page_start,
        "page_end": page_end
    }
    
    # Prepend hierarchical context to the chunk text for improved vector retrieval
    context_prefix = f"Source: {source_name}"
    if chapter:
        context_prefix += f" | Chapter: {chapter}"
    if section:
        context_prefix += f" | Section: {section}"
    if subsection:
        context_prefix += f" | Subsection: {subsection}"
    context_prefix += f" | Pages: {page_start}-{page_end}\n\n"
    
    return {
        "text": context_prefix + text.strip(),
        "metadata": metadata
    }

def parse_pdf_to_chunks(filepath: str, source_name: str, max_words: int = 500, pages: List[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a PDF file, converts page-by-page to Markdown using pymupdf4llm,
    and applies semantic Markdown-heading based chunking.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found at {filepath}")
        
    print(f"Converting PDF to Markdown page chunks for: {filepath}...")
    # Get page chunks with pymupdf4llm
    pages_data = pymupdf4llm.to_markdown(filepath, page_chunks=True, pages=pages)
    print(f"Successfully converted {len(pages_data)} pages.")
    
    chunks = []
    current_header_path = []
    current_chunk_lines = []
    
    # Initialize page tracking based on the first page's metadata
    first_page_num = pages_data[0]["metadata"].get("page_number", 1) if pages_data else 1
    page_start = first_page_num
    page_end = first_page_num
    
    for page in pages_data:
        p_num = page["metadata"].get("page_number", 1)
        text = page.get("text", "")
        
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            # Match Markdown headings (e.g., # Heading, ## Heading)
            match = re.match(r'^(#+)\s+(.*)$', stripped)
            if match:
                # If we have accumulated lines, output them before starting the new heading
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        chunks.append(
                            create_chunk(
                                text=chunk_text,
                                source_name=source_name,
                                header_path=current_header_path,
                                page_start=page_start,
                                page_end=page_end
                            )
                        )
                    current_chunk_lines = []
                
                # Update current header state
                level = len(match.group(1))
                title = clean_title(match.group(2))
                current_header_path = update_header_path(current_header_path, level, title)
                page_start = p_num
                page_end = p_num
                
                current_chunk_lines.append(line)
            else:
                if stripped:
                    current_chunk_lines.append(line)
                    page_end = p_num
                    
                    # Split if chunk reaches size threshold
                    word_count = sum(len(l.split()) for l in current_chunk_lines)
                    if word_count >= max_words:
                        chunk_text = "\n".join(current_chunk_lines).strip()
                        if chunk_text:
                            chunks.append(
                                create_chunk(
                                    text=chunk_text,
                                    source_name=source_name,
                                    header_path=current_header_path,
                                    page_start=page_start,
                                    page_end=page_end
                                )
                            )
                        current_chunk_lines = []
                        page_start = p_num
                        
    # Flush remaining text
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append(
                create_chunk(
                    text=chunk_text,
                    source_name=source_name,
                    header_path=current_header_path,
                    page_start=page_start,
                    page_end=page_end
                )
            )
            
    return chunks
