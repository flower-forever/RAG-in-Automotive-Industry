# Full-Stack RAG System Design Spec
Date: 2026-06-23
Topic: Fullstack RAG with Next.js, Tailwind v3, and FastAPI

## 1. Goal
Transition the existing CLI-based RAG application into a full-stack Web Application suitable for a Hackathon (Tier 3), featuring a rich UI, file uploads, and advanced RAG features (CoT, Multi-query, Self-correction).

## 2. Architecture & Data Flow
- **Frontend**: Next.js (App Router) + Tailwind CSS v3.
- **Backend**: FastAPI wrapping the existing Python RAG modules (`retrieval.py`, `generation.py`).
- **File Upload (Synchronous)**: 
  - Frontend limits: Max 3 files simultaneously, max 5MB per file, PDF/TXT only.
  - Backend receives files, saves them, and synchronously calls `build_index`.
  - Frontend displays a "Building Index..." loading state until completion.

## 3. Visual Design & Theme (Frontend Design)
The application will support both Light (default) and Dark themes with an "Industrial Cybersecurity" aesthetic.

### Typography
- **Display/UI**: `JetBrains Mono` (for stats, tags, code-like elements).
- **Body**: `Inter` (for readable chat bubbles and document content).

### Palette (Light Mode - Primary)
- **Background**: `#F8FAFC` (Slate 50) - Clean, clinical, modern workspace.
- **Surface**: `#FFFFFF` (White) - Card and chat bubble backgrounds.
- **Primary/Action**: `#0284C7` (Sky 600) - Actionable elements.
- **Text**: `#0F172A` (Slate 900) - High readability.
- **Success/High Confidence**: `#10B981` (Emerald 500).
- **Alert/Low Confidence**: `#E11D48` (Rose 600).

### Palette (Dark Mode - Secondary)
- **Background**: `#0B1120` (Dark Space).
- **Surface**: `#1E293B` (Slate 800).
- **Primary/Action**: `#06B6D4` (Cyber Cyan).

### Signature Element
- **Thinking Topology Drawer**: When the AI processes a query, an animated "pulse" indicates thinking. Once complete, a collapsible accordion drawer allows users to expand and view the LLM's Chain of Thought (analysis, critique, synthesis).

## 4. Layout Overview
- **Header**: Logo, title, Theme Toggle (Light/Dark), and "Upload Files" button.
- **Left Sidebar**: 
  - Filter Controls (Vendor, Severity, Source).
  - System Status (Index Chunk Count, Model Status).
- **Main Chat Area**: 
  - Chat history.
  - LLM Responses with "Confidence Score" badges.
  - Collapsible CoT Drawer under the response.
  - Clickable Source Citations linking to metadata.

## 5. Security & Resource Constraints
- Strictly enforced upload limits (3 files, 5MB) on both client and server to prevent resource exhaustion on local machines.
- Synchronous indexing provides immediate failure/success feedback without background task complexity.
