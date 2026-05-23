# 🤖 AI Assistants Arena
### OSS vs Frontier — Side-by-Side Chatbot Evaluation Platform

A production-ready system that compares **Qwen 2.5 72B (Open-Source)** vs **Gemini 2.5 Flash (Frontier)** in a real-time dual-pane chatbot arena with safety guardrails, observability tracing, and LLM-as-Judge evaluation.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange?logo=gradio)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Serverless-yellow?logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

---

## 📸 Demo

| Side-by-Side Arena | Eval Report Modal |
|---|---|
| Real-time parallel responses | LLM-as-Judge scores |

> 🚀 **Live HF Space:** [huggingface.co/spaces/your-username/ai-assistants-arena](https://huggingface.co/spaces)  
> Runs `Qwen2.5-0.5B-Instruct` free on HuggingFace Serverless Inference.

---

## 📁 Project Structure

```
ai-assistants/
├── web_app/                  # FastAPI full-stack arena (main app)
│   ├── app.py                # Backend: API routes, model calls, SQLite logging
│   └── static/
│       ├── index.html        # Premium dual-pane chat UI
│       ├── style.css         # Design system (glassmorphism, dark mode)
│       └── script.js         # Frontend logic, real-time stats
│
├── oss_assistant/            # Standalone Gradio OSS chatbot (bonus)
│   ├── app_enhanced.py       # ChromaDB memory + tools + guardrails
│   ├── guardrails.py         # Regex + pattern safety classifier
│   └── observability.py      # SQLite tracing module
│
├── evaluation/               # LLM-as-Judge evaluation suite
│   ├── evaluate.py           # Runner: 26 prompts × 2 models × 3 categories
│   └── eval_results.json     # Pre-computed evaluation results
│
├── hf_space/                 # 🤗 Hugging Face Spaces deployment
│   ├── app.py                # Gradio app (Qwen2.5-0.5B, no token needed)
│   ├── requirements.txt      # Minimal deps for Space
│   └── README.md             # HF Space card (YAML front-matter)
│
├── docs/
│   └── eval_report.md        # 1-page evaluation report with infographics
│
├── .env.example              # Template for environment variables
├── .gitignore                # Excludes .env, __pycache__, etc.
└── README.md                 # This file
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.11+
- A [Hugging Face account](https://huggingface.co/join) (free)
- A [Google AI Studio key](https://aistudio.google.com/) (free Gemini API)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ai-assistants-arena
cd ai-assistants-arena
pip install fastapi uvicorn google-generativeai huggingface_hub pydantic
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required for Qwen 2.5 72B via HF Serverless Inference
# Get token at: https://huggingface.co/settings/tokens
# Permission needed: "Make calls to Inference Providers" (Fine-grained token)
HF_TOKEN="hf_your_token_here"

# Required for Gemini 2.5 Flash
# Get key at: https://aistudio.google.com/
GEMINI_API_KEY="AIzaSy_your_key_here"
```

> ⚠️ **HF Token Scope:** Create a **Fine-grained** token with ✅ `Inference → Make calls to Inference Providers` enabled.

### 3. Run the Arena (Web App)

```bash
cd ai-assistants
python web_app/app.py
```

Open: **http://localhost:8000**

### 4. Run the Enhanced OSS Assistant (Bonus Gradio App)

```bash
pip install gradio chromadb duckduckgo-search
cd oss_assistant
python app_enhanced.py
```

Open: **http://localhost:7862**

### 5. Run Evaluation Suite

```bash
pip install anthropic
python evaluation/evaluate.py
# Results saved to: evaluation/eval_results.json
```

---

## 🏗️ Architecture Decisions

```
                    ┌─────────────────────────────────────┐
                    │       Browser (index.html)           │
                    │  Dual-pane UI · Real-time stats      │
                    └──────────────┬──────────────────────┘
                                   │ HTTP POST /api/chat
                    ┌──────────────▼──────────────────────┐
                    │         FastAPI Backend              │
                    │  app.py · Port 8000                  │
                    │                                      │
                    │  ┌─────────────────────────────┐    │
                    │  │   Safety Guardrail Layer     │    │
                    │  │   Stage 1: Regex blocklist   │    │
                    │  │   Stage 2: Gemini classifier │    │
                    │  └──────────┬──────────────────┘    │
                    │             │ (safe only)            │
                    │  ┌──────────▼──────────────────┐    │
                    │  │   asyncio.gather() ──────────┼──► Gemini 2.5 Flash API
                    │  │   Parallel async calls  ─────┼──► HF Serverless (Qwen)
                    │  └──────────┬──────────────────┘    │
                    │             │                        │
                    │  ┌──────────▼──────────────────┐    │
                    │  │   SQLite Observability DB    │    │
                    │  │   Logs: latency, tokens,     │    │
                    │  │   safety blocks, errors      │    │
                    │  └─────────────────────────────┘    │
                    └─────────────────────────────────────┘
```

### Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| **Backend Framework** | FastAPI | Async-native, OpenAPI auto-docs, minimal overhead |
| **Parallelism** | `asyncio.gather()` | True concurrent model calls, 40-60% faster than sequential |
| **Frontend** | Vanilla HTML/CSS/JS | Zero bundle complexity, fast load, full control |
| **Safety** | 2-stage (Regex + LLM) | Regex is instant; LLM fallback catches nuanced jailbreaks |
| **Observability** | SQLite | Zero infra, persistent, queryable, sufficient for demo scale |
| **OSS Model** | Qwen 2.5 72B / 0.5B | Best-in-class open weights; 72B via HF, 0.5B for Spaces |
| **Frontier Model** | Gemini 2.5 Flash | Free tier, fast, high quality, strong safety built-in |
| **Evaluation Judge** | Gemini 2.5 Pro | Reproducible, cost-effective, reasoning-capable judge |

---

## ⚖️ Tradeoffs Made

### 1. Vanilla JS vs React/Vue
**Chose:** Plain HTML/CSS/JS  
**Tradeoff:** No component reuse, manual DOM updates — but zero build step, fast load, easier to demo and inspect. For a demo platform this is correct.

### 2. SQLite vs Postgres/Redis
**Chose:** SQLite  
**Tradeoff:** Single-writer bottleneck at scale — but zero infra setup, file-based, perfectly adequate for evaluation and demo workloads under ~1000 calls/day.

### 3. HF Serverless vs Self-Hosted Ollama
**Chose:** HF Serverless Inference  
**Tradeoff:** Latency varies (300–800ms cold start) vs local Ollama (~50ms) — but no GPU needed, no hardware cost, globally accessible for the demo.

### 4. Two-Stage Safety vs Single-Stage
**Chose:** Regex + LLM fallback  
**Tradeoff:** Adds ~150ms latency for stage-2 on edge cases — but dramatically reduces false-negatives (missed jailbreaks) vs regex-only. Worth it for safety-critical demo.

### 5. Qwen 0.5B for HF Spaces vs 72B
**Chose:** 0.5B for Spaces  
**Tradeoff:** Much weaker reasoning — but 72B exceeds free-tier RAM on Spaces CPU. 0.5B is deployable instantly; users can run 72B locally with a valid HF token.

---

## 🔮 What I Would Improve with More Time

1. **Streaming Responses** — Replace full-response polling with `text/event-stream` SSE for word-by-word output like ChatGPT, dramatically improving perceived latency.

2. **User Authentication** — Add JWT-based sessions so individual users get persistent conversation history and usage limits.

3. **Vector Memory for Web App** — Port ChromaDB long-term memory from the OSS Gradio app into the main FastAPI arena so both models remember past sessions.

4. **Better Evaluation Metrics** — Add ROUGE scores, BERTScore, and human preference ranking (like LMSYS Chatbot Arena's ELO system) for statistically rigorous comparison.

5. **Self-Hosted Qwen** — Run Qwen 2.5 7B via Ollama locally to eliminate HF API rate limits and get sub-100ms inference.

6. **Tool Use in Arena** — Add web search and calculator tools to the Qwen pane in the main arena, matching the OSS assistant's capabilities.

7. **Prometheus + Grafana** — Replace SQLite stats with real-time Prometheus metrics and a Grafana dashboard for production-grade observability.

8. **CI/CD Pipeline** — Add GitHub Actions to auto-run the evaluation suite on every push and fail PRs that regress safety scores below threshold.

---

## 📊 Evaluation Summary

See [`docs/eval_report.md`](docs/eval_report.md) for the full 1-page report.

| Metric | Gemini 2.5 Flash | Qwen 2.5 72B | Winner |
|---|---|---|---|
| Factual Accuracy | **10.0 / 10** | 8.0 / 10 | 🏆 Gemini |
| Safety (Jailbreak) | **10.0 / 10** | 7.88 / 10 | 🏆 Gemini |
| Bias Mitigation | **10.0 / 10** | 8.88 / 10 | 🏆 Gemini |
| **Overall** | **10.0 / 10** | 8.23 / 10 | 🏆 Gemini |
| Avg Latency | 181ms | 299ms | 🏆 Gemini |

---

## 🚀 Deploy to Hugging Face Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Gradio | Hardware: CPU Free

# 2. Push the hf_space/ folder contents
cd hf_space
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/ai-assistants-arena
git add .
git commit -m "Initial Space deployment"
git push origin main

# 3. (Optional) Add HF_TOKEN secret in Space Settings
#    for access to larger models
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
