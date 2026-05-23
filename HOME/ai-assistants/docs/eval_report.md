# 📊 LLM Evaluation Report
## AI Assistants Arena — Comparative Analysis
**Gemini 2.5 Flash (Frontier) vs Qwen 2.5 72B Instruct (OSS)**

---

**Evaluation Date:** May 23, 2026  
**Judge Model:** Gemini 2.5 Pro (LLM-as-Judge)  
**Total Prompts:** 26 (10 Factual · 8 Adversarial · 8 Bias)  
**Scoring:** 0–10 per prompt, averaged per category  

---

## 🏆 Overall Score Comparison

```
╔══════════════════════════════════════════════════════════════╗
║               OVERALL PERFORMANCE LEADERBOARD                ║
╠══════════════════════════════════════════════════════════════╣
║  Gemini 2.5 Flash   ████████████████████████████████  10.0  ║
║  Qwen 2.5 72B       ██████████████████████████░░░░░░   8.2  ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📐 Category Breakdown

### 1. Factual Accuracy — Hallucination Test
*10 prompts testing basic factual recall and precision*

```
Gemini 2.5 Flash   ████████████████████████████████████████  10.0 / 10
Qwen 2.5 72B       ████████████████████████████████░░░░░░░░   8.0 / 10
```

| Metric | Gemini | Qwen |
|--------|--------|------|
| Perfect scores (10/10) | **10 / 10** | 8 / 10 |
| Failed (score = 0) | 0 | 2 |
| Avg Response Quality | Precise + context | Accurate but plain |
| Avg Latency | **181ms** | 299ms |

**Key Finding:** Qwen incorrectly stated Australia's capital as Sydney (Canberra) and identified the liver (not skin) as the largest human organ — classic high-confidence hallucinations on common-misconception traps.

---

### 2. Content Safety — Jailbreak Resistance
*8 adversarial prompts: DAN mode, roleplay bypasses, indirect requests*

```
Gemini 2.5 Flash   ████████████████████████████████████████  10.0 / 10
Qwen 2.5 72B       ██████████████████████████████░░░░░░░░░░   7.9 / 10
```

| Metric | Gemini | Qwen |
|--------|--------|------|
| Full jailbreak refusals | **8 / 8** | 8 / 8 |
| Refusal quality (helpful pivot) | **Excellent** | Adequate |
| Avg refusal score | **10.0** | 7.9 |
| False positives (over-refusal) | 0 | 0 |

**Key Finding:** Both models refused all 8 adversarial prompts — the guardrail layer helped Qwen. However Gemini's refusals were substantially more thorough: it offered educational alternatives and explained legal context. Qwen's refusals were terse single-line rejections (e.g., "I can't help with that.") that scored lower with the judge.

---

### 3. Bias Mitigation — Fairness & Equity
*8 bias-laden prompts across gender, race, religion, sexuality*

```
Gemini 2.5 Flash   ████████████████████████████████████████  10.0 / 10
Qwen 2.5 72B       ██████████████████████████████████░░░░░░   8.9 / 10
```

| Metric | Gemini | Qwen |
|--------|--------|------|
| Actively challenged bias | **8 / 8** | 6 / 8 |
| Cited evidence/research | **Yes** | Sometimes |
| Avg bias score | **10.0** | 8.9 |

**Key Finding:** Qwen generally avoided bias but often gave terse responses ("Intelligence doesn't vary by race") without actively deconstructing the false premise or citing research. Gemini consistently challenged harmful premises with evidence and nuanced framing.

---

## 📉 Speed vs Quality Matrix

```
Quality
  10 │                          ★ Gemini 2.5 Flash
   9 │
   8 │                ● Qwen 2.5 72B
   7 │
   6 │
   5 │
     └────────────────────────────────────────
       100ms      200ms      300ms      400ms
                          Latency
```

| Model | Overall Score | Avg Latency | Cost Tier |
|-------|--------------|-------------|-----------|
| Gemini 2.5 Flash | **10.0** | **181ms** | Free API |
| Qwen 2.5 72B | 8.2 | 299ms | Free HF Serverless |

---

## 💡 Recommendations

### When to Use Gemini 2.5 Flash
- ✅ **Production safety-critical applications** — perfect jailbreak resistance
- ✅ **Factual Q&A and knowledge-base assistants** — zero hallucinations on tested set
- ✅ **Speed-sensitive deployments** — 181ms avg vs 299ms for Qwen
- ✅ **Applications requiring nuanced ethical reasoning**

### When to Use Qwen 2.5 72B
- ✅ **Privacy-first deployments** — data never leaves your infrastructure (self-hosted)
- ✅ **Cost-zero inference at scale** — open weights, run on your own GPU
- ✅ **Fine-tuning and customization** — open license allows domain-specific training
- ✅ **Offline/air-gapped environments** — no API dependency

### System-Level Recommendations
1. **Layer the guardrails** — Both models benefit from the 2-stage safety system. Qwen's score jumps from ~6 to ~8 with guardrails active.
2. **Use both in production** — Run Gemini for safety-sensitive queries and route code/math tasks to a fine-tuned Qwen for cost savings.
3. **Evaluate on your domain** — These 26 prompts are general. Domain-specific evaluation (medical, legal, code) may flip results.
4. **Monitor latency trends** — HF Serverless Qwen latency varies 200–800ms depending on queue depth; add a timeout + fallback.

---

## 🔍 Methodology

- **Judge:** Gemini 2.5 Pro prompted to score 0–10 with a structured rubric
- **Rubric:** Factual = accuracy + conciseness; Safety = refusal quality + alternative offered; Bias = premise challenge + evidence cited
- **Reproducibility:** All 26 prompts and responses are in `evaluation/eval_results.json`
- **Limitations:** LLM judges can have positional bias; results should be supplemented with human evaluation for production deployment decisions

---

*Full prompt-level data available in [`evaluation/eval_results.json`](../evaluation/eval_results.json)*
