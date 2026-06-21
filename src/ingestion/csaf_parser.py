import os
import json
import glob
from typing import List, Dict, Any, Optional

def recursive_extract_products(node: Dict[str, Any], product_map: Dict[str, Dict[str, str]], current_vendor: str = "") -> None:
    """
    Recursively parse the product_tree branches to map product_id to product_name and vendor.
    """
    category = node.get("category", "")
    name = node.get("name", "")
    
    if category == "vendor":
        current_vendor = name
        
    if "product" in node:
        prod_info = node["product"]
        p_id = prod_info.get("product_id")
        p_name = prod_info.get("name")
        if p_id:
            product_map[p_id] = {
                "name": p_name,
                "vendor": current_vendor or "Unknown Vendor"
            }
            
    for branch in node.get("branches", []):
        recursive_extract_products(branch, product_map, current_vendor)

def parse_single_csaf(filepath: str) -> List[Dict[str, Any]]:
    """
    Parses a single CSAF v2.0 JSON file and returns a list of semantic chunks.
    Each chunk is a dict: {"text": str, "metadata": dict}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    doc = data.get("document", {})
    tracking = doc.get("tracking", {})
    advisory_id = tracking.get("id", "Unknown-ID")
    title = doc.get("title", "Unknown Title")
    release_date = tracking.get("current_release_date", "").split("T")[0]
    publisher = doc.get("publisher", {}).get("name", "CISA")
    
    # Extract metadata fields from notes
    notes = doc.get("notes", [])
    summary_text = next((n.get("text", "") for n in notes if n.get("category") == "summary"), "")
    sector = next((n.get("text", "") for n in notes if n.get("category") == "other" and n.get("title") == "Critical infrastructure sectors"), "")
    hq = next((n.get("text", "") for n in notes if n.get("category") == "other" and n.get("title") == "Company headquarters location"), "")
    
    # Extract products mapping from product_tree
    product_map = {}
    if "product_tree" in data:
        for branch in data["product_tree"].get("branches", []):
            recursive_extract_products(branch, product_map)
            
    all_vendors = sorted(list(set(info["vendor"] for info in product_map.values())))
    all_products = sorted(list(set(info["name"] for info in product_map.values())))
    
    # Compute maximum severity for the overview metadata
    max_severity = "LOW"
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    for vuln in data.get("vulnerabilities", []):
        for score_obj in vuln.get("scores", []):
            cvss3 = score_obj.get("cvss_v3", {})
            sev = cvss3.get("baseSeverity", "LOW").upper()
            if severity_rank.get(sev, 0) > severity_rank.get(max_severity, 0):
                max_severity = sev

    chunks = []
    
    # ----------------------------------------------------
    # Chunk 1: Advisory Overview Chunk
    # ----------------------------------------------------
    overview_text = f"# CISA ICS Advisory: {title}\n"
    overview_text += f"- **Advisory ID**: {advisory_id}\n"
    overview_text += f"- **Release Date**: {release_date}\n"
    overview_text += f"- **Publisher**: {publisher}\n"
    if sector:
        overview_text += f"- **Critical Infrastructure Sector**: {sector}\n"
    if hq:
        overview_text += f"- **Headquarters Location**: {hq}\n"
    if all_vendors:
        overview_text += f"- **Vendor**: {', '.join(all_vendors)}\n"
    if all_products:
        overview_text += f"- **Affected Products**:\n"
        for p in all_products:
            overview_text += f"  - {p}\n"
            
    if summary_text:
        overview_text += f"\n## Advisory Summary\n{summary_text}\n"
        
    overview_metadata = {
        "source": "CISA_CSAF",
        "advisory_id": advisory_id,
        "title": title,
        "vendor": ", ".join(all_vendors) if all_vendors else "Unknown",
        "products": "; ".join(all_products) if all_products else "Unknown",
        "date": release_date,
        "severity": max_severity,
        "sector": sector or "Unknown",
        "chunk_type": "overview"
    }
    chunks.append({"text": overview_text.strip(), "metadata": overview_metadata})
    
    # ----------------------------------------------------
    # Chunk 2: Individual Vulnerability Chunks
    # ----------------------------------------------------
    for vuln in data.get("vulnerabilities", []):
        cve_id = vuln.get("cve", "Unknown-CVE")
        cwe_info = vuln.get("cwe", {})
        cwe_id = cwe_info.get("id", "")
        cwe_name = cwe_info.get("name", "")
        
        vuln_notes = vuln.get("notes", [])
        vuln_summary = next((n.get("text", "") for n in vuln_notes if n.get("category") == "summary"), "")
        
        # Get severity score
        vuln_score = 0.0
        vuln_severity = "LOW"
        vuln_vector = ""
        for score_obj in vuln.get("scores", []):
            cvss3 = score_obj.get("cvss_v3", {})
            if cvss3:
                vuln_score = cvss3.get("baseScore", vuln_score)
                vuln_severity = cvss3.get("baseSeverity", vuln_severity).upper()
                vuln_vector = cvss3.get("vectorString", vuln_vector)
                break  # Take the first CVSS v3 score
                
        # Affected products for this vulnerability
        affected_pids = vuln.get("product_status", {}).get("known_affected", [])
        affected_prods = []
        affected_vendors = []
        for pid in affected_pids:
            if pid in product_map:
                affected_prods.append(product_map[pid]["name"])
                affected_vendors.append(product_map[pid]["vendor"])
                
        vuln_vendors_str = ", ".join(sorted(list(set(affected_vendors)))) or ", ".join(all_vendors)
        vuln_prods_str = "; ".join(sorted(list(set(affected_prods)))) or "; ".join(all_products)
        
        vuln_text = f"# CISA ICS Advisory {advisory_id} - Vulnerability Details\n"
        vuln_text += f"- **Vulnerability CVE**: {cve_id}\n"
        if cwe_id or cwe_name:
            vuln_text += f"- **CWE**: {cwe_id}: {cwe_name}\n"
        vuln_text += f"- **Severity**: {vuln_severity} (CVSS Score: {vuln_score})\n"
        if vuln_vector:
            vuln_text += f"- **CVSS Vector**: {vuln_vector}\n"
        if vuln_vendors_str:
            vuln_text += f"- **Vendor**: {vuln_vendors_str}\n"
        if vuln_prods_str:
            vuln_text += f"- **Affected Products**: {vuln_prods_str}\n"
            
        if vuln_summary:
            vuln_text += f"\n## Description\n{vuln_summary}\n"
            
        # Extract remediations for this vulnerability
        remediations = vuln.get("remediations", [])
        if remediations:
            vuln_text += f"\n## Remediations and Mitigations\n"
            for rem in remediations:
                category = rem.get("category", "remediation")
                details = rem.get("details", "")
                url = rem.get("url", "")
                vuln_text += f"- **Type**: {category.replace('_', ' ').title()}\n"
                if details:
                    vuln_text += f"  - **Details**: {details}\n"
                if url:
                    vuln_text += f"  - **Reference URL**: {url}\n"
                    
        vuln_metadata = {
            "source": "CISA_CSAF",
            "advisory_id": advisory_id,
            "title": title,
            "vendor": vuln_vendors_str or "Unknown",
            "products": vuln_prods_str or "Unknown",
            "date": release_date,
            "severity": vuln_severity,
            "cve": cve_id,
            "cwe_id": cwe_id,
            "chunk_type": "vulnerability"
        }
        chunks.append({"text": vuln_text.strip(), "metadata": vuln_metadata})
        
    # ----------------------------------------------------
    # Chunk 3: Individual Recommended Practice Chunks (Fine-grained)
    # ----------------------------------------------------
    rec_notes = [n for n in notes if n.get("category") == "general" and n.get("title") == "Recommended Practices"]
    for i, note in enumerate(rec_notes):
        text_content = note.get("text", "").strip()
        if not text_content:
            continue
            
        rec_text = f"# CISA ICS Advisory {advisory_id} - Recommended Practice\n"
        if all_vendors:
            rec_text += f"- **Vendor**: {', '.join(all_vendors)}\n"
        if all_products:
            rec_text += f"- **Affected Products**: {'; '.join(all_products)}\n"
        rec_text += f"\n## Recommendation\n{text_content}\n"
        
        rec_metadata = {
            "source": "CISA_CSAF",
            "advisory_id": advisory_id,
            "title": title,
            "vendor": ", ".join(all_vendors) if all_vendors else "Unknown",
            "products": "; ".join(all_products) if all_products else "Unknown",
            "date": release_date,
            "severity": max_severity,
            "chunk_type": "recommendation"
        }
        chunks.append({"text": rec_text.strip(), "metadata": rec_metadata})
        
    return chunks

def parse_all_csaf_dir(csaf_dir: str) -> List[Dict[str, Any]]:
    """
    Parses all JSON files in the specified directory and returns a aggregated list of chunks.
    """
    all_chunks = []
    pattern = os.path.join(csaf_dir, "*.json")
    files = glob.glob(pattern)
    for filepath in files:
        all_chunks.extend(parse_single_csaf(filepath))
    return all_chunks
