# RAG-in-Automotive-Industry: SecureOps Assistant for ICS Cybersecurity

A Retrieval-Augmented Generation (RAG) system built for the **AAI Tech Talks Hackathon 2026** (University of Warwick, MSc Applied AI). This assistant is designed to help security operations teams query and analyze industrial control systems (ICS) and operational technology (OT) vulnerabilities, security frameworks, and threat matrices.

---

## Data Sources

The project utilizes the following datasets and reference materials:

1. **CISA CSAF Security Advisories**: Machine-readable ICS vulnerability advisories in CSAF v2.0 JSON format.
2. **NIST SP 800-82 Rev. 3**: Guide to Operational Technology (OT) Security.
3. **NIST Cybersecurity Framework (CSF) 2.0**: The industry standard for managing cybersecurity risk.
4. **MITRE ATT&CK® for ICS (v19.1)**: Adversary tactics, techniques, and procedures (TTPs) targeting industrial control systems.

---

## Dataset Citation & References

To ensure academic and professional reproducibility, the CISA CSAF dataset and other reference materials should be cited as follows:

#### CISA CSAF (Common Security Advisory Framework) Dataset
The vulnerability advisories are sourced from the official Cybersecurity and Infrastructure Security Agency (CISA) CSAF repository.

* **APA:** Cybersecurity and Infrastructure Security Agency (CISA). (2026). *Common Security Advisory Framework (CSAF) ICS Advisories Dataset*. GitHub Repository. Retrieved from https://github.com/cisagov/CSAF

* **BibTeX (for LaTeX)**
```bibtex
@misc{cisa2026csaf,
  author       = {{Cybersecurity and Infrastructure Security Agency (CISA)}},
  title        = {{Common Security Advisory Framework (CSAF) ICS Advisories Dataset}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/cisagov/CSAF}},
  note         = {Accessed: 2026-06-18}
}
```

---

#### **NIST SP 800-82 Rev. 3**
* **APA:** Stouffer, K., Pillitteri, V., Lightman, S., Abrams, M., & Hahn, A. (2023). *Guide to Operational Technology (OT) Security* (NIST Special Publication 800-82 Rev. 3). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-82r3
* **BibTeX:**
  ```bibtex
  @techreport{nist80082r3,
    author      = {Keith Stouffer and Victoria Pillitteri and Suzanne Lightman and Marshall Abrams and Adam Hahn},
    title       = {{Guide to Operational Technology (OT) Security}},
    institution = {National Institute of Standards and Technology (NIST)},
    year        = {2023},
    number      = {NIST SP 800-82 Rev. 3},
    doi         = {10.6028/NIST.SP.800-82r3}
  }
  ```

#### **NIST CSF 2.0**
* **APA:** National Institute of Standards and Technology. (2024). *The NIST Cybersecurity Framework (CSF) 2.0*. https://doi.org/10.6028/NIST.CSWP.29
* **BibTeX:**
  ```bibtex
  @techreport{nistcsf2,
    author      = {{National Institute of Standards and Technology (NIST)}},
    title       = {{The NIST Cybersecurity Framework (CSF) 2.0}},
    institution = {National Institute of Standards and Technology (NIST)},
    year        = {2024},
    number      = {NIST CSWP 29},
    doi         = {10.6028/NIST.CSWP.29}
  }
  ```

#### **MITRE ATT&CK® for ICS (v19.1)**
* **APA:** The MITRE Corporation. (2025). *MITRE ATT&CK for Industrial Control Systems (v19.1)*. https://attack.mitre.org/matrices/ics
* **BibTeX:**
  ```bibtex
  @misc{mitreattackics2025,
    author       = {{The MITRE Corporation}},
    title        = {{MITRE ATT\&CK for Industrial Control Systems (v19.1)}},
    year         = {2025},
    howpublished = {\url{https://attack.mitre.org/matrices/ics}},
    note         = {Accessed: 2026-06-18}
  }
  ```

---

## License

This project is licensed under the terms in the LICENSE file in this repository.

---

## Quick Start & New Commands

Follow these steps from the project root to run the interactive SecureOps demo and use the new features added in this version (parallel rewrite retrieval, device fallback, and rewrite visibility).

1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2) (Optional) Configure Gemini API key for generation

PowerShell:
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```
CMD:
```cmd
set GEMINI_API_KEY=your_api_key_here
```

3) (Optional) Force CPU-only model loading (avoids CUDA device errors)

PowerShell:
```powershell
$env:RAG_DEVICE="cpu"
```

4) Run the CLI

```powershell
python src/cli.py
```

5) Common CLI commands

- `rebuild quick`  — quick index rebuild (samples only) for fast demos
- `rebuild`        — full index rebuild (CSAF JSONs + NIST PDFs)
- `ask <question>` — runs retrieval+generation; note: `ask` automatically generates rewrite candidates, performs parallel retrieval over them, and selects the best candidate by reranker score
- `rewrite <question>` — show rewrite/paraphrase candidates without invoking retrieval/generation
- `filter ...`     — set metadata filters (vendor, severity, source)
- `sources`        — print index statistics
- `exit` / `quit`  — exit the CLI

Notes:
- Parallel candidate retrieval is enabled for `ask` to lower latency when multiple rewrite candidates are generated.
- Device selection: the code checks environment variables `RAG_DEVICE` / `TORCH_DEVICE` / `PYTORCH_DEVICE` and falls back to CPU if CUDA is unavailable or loading fails.
- The original query is always included among rewrite candidates as a fallback.

If you want, I can add a one-line demo script that runs `rebuild quick` then a sample `ask` and times the retrieval to produce a quick benchmark.
