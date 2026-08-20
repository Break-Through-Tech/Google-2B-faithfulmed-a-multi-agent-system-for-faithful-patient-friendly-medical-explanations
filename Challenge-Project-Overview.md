# FaithfulMed: A Multi-Agent System for Faithful, Patient-Friendly Medical Explanations

**Company / Org:** Google
**Challenge Advisor:** Sarita Anand Joshi, sarita.ritu@gmail.com
**Program:** Break Through Tech AI Studio — Fall 2026

---

## 🏢 About Google
Google is a global technology leader focused on organizing the world's information and making it universally accessible and useful. Health information is one of the hardest categories to make truly *accessible*: the source material (clinical notes, lab reports, discharge instructions) is written for clinicians, not patients. This project sits at the intersection of two Google-relevant themes — trustworthy generative AI and equitable access to information — and is built on Google's own open stack (Gemini, Gemma, and the Agent Development Kit).

---

## 🎯 The Challenge

### Project Summary
In this project, you will use publicly available de-identified clinical text (MTSamples transcribed medical reports, Synthea synthetic patient records, MedQuAD consumer health Q&A, and the PLABA plain-language adaptation corpus), together with the MedAESQA evidence-supported medical QA dataset, and a **multi-agent LLM architecture** — combining retrieval-augmented generation (RAG), prompt engineering, source-grounded faithfulness verification, and iterative agent-based refinement using Google's **Gemini** and open-weight **Gemma** models (plus one external open model for comparison) — to build a system in which specialized agents (**Extractor, Simplifier, Verifier, Refiner, Readability**) collaborate to produce patient-friendly explanations of clinical documents that are simultaneously **easy to read** and **provably faithful to the source**.

This addresses the well-documented **health-literacy gap** — where patients routinely misunderstand discharge instructions, lab results, and care plans written above their reading level — while tackling the central blocker to deploying LLMs for patient-facing use: **hallucinated or unfaithful information** that could harm patients if trusted.

### The Core Research Question
> Does a pipeline of specialized agents produce explanations that are *both* more readable *and* more faithful than a single well-prompted LLM — and **which agents actually matter** for each goal?

A single well-prompted Gemini call is a strong baseline. Your job is to show, with evidence, where a multi-agent design beats it and where it doesn't.

### Success Criteria
A successful December outcome includes:

**Quantitative system performance**
- **Readability:** ≥80% of outputs at ≤8th-grade reading level (Flesch-Kincaid).
- **Faithfulness:** ≥85% factual fidelity on the human-annotated test set, verified against human labels.
- **Hallucination rate:** <10% of outputs contain a clinically meaningful unsupported claim.
- **Multi-agent vs. single-agent baseline:** ≥15% absolute improvement in faithfulness score.

**Verifier agent quality**
- Verifier agrees with human annotators ≥80% of the time on faithfulness labels (target Cohen's κ ≥ 0.6).

**Ablation analysis (research contribution)**
- A complete ablation table showing the marginal contribution of each agent — answers "which agents matter most for faithfulness vs. readability?"

**Comparative analysis**
- Published leaderboard across **≥3 models spanning Gemini (API) and open weights (the Gemma family + one external open model)**, showing faithfulness, readability, latency, cost, **and openness** tradeoffs.

**Working artifacts**
- Public GitHub repo with reproducible code and one notebook per agent.
- Hosted demo (Hugging Face Spaces or Cloud Run) showing per-agent traces and evidence attributions.
- Technical report (~10 pages) including the ablation study, plus a final team presentation.
- Portfolio-ready writeup.
- Each fellow can speak to their owned agent in interviews and on LinkedIn (e.g., *"I designed and validated the Faithfulness Verifier agent against human-annotated medical text"*).

### Stretch Goals
For teams that progress quickly:
- **Multilingual extension** — add Spanish generation with a parallel Verifier (biggest equity impact; Spanish-speaking US populations face larger health-literacy gaps).
- **Personalization layer** — adjustable target reading level and tone (formal / conversational) through the Readability agent.
- **Agent-debate variant** — replace the single Verifier with a two-agent debate; compare against the single-Verifier ablation.
- **Bias & fairness audit** — does the pipeline perform worse on reports involving non-English names, rare conditions, or specific demographic markers? Run a structured audit and publish findings.
- **LoRA fine-tuning experiment** — fine-tune a small open model (e.g., Gemma-2-2B) on patient-friendly explanation pairs and compare against prompt-only baselines.
- **Domain expansion** — apply the pipeline to a high-impact subdomain (oncology discharge summaries, post-partum care instructions, pediatric medication labels).
- **Tool-augmented agents** — give the Extractor a UMLS lookup tool and the Verifier a DrugBank lookup tool; measure whether tools improve verification accuracy.
- **Multimodal clinical data integration** — use Gemini's native multimodality to process visual medical assets (e.g., radiology images or ECG waveforms) alongside textual records for joint explanations.
- **Real-time guardrail middleware** — deploy the Verifier as an interceptor (via ADK callbacks) that flags unverified clinical claims *before* a final answer is shown.
- **Counterfactual explanation visualizer** — an interactive module showing how changing a lab value or history item alters the multi-agent reasoning path.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Foundations & Single-Agent Baseline | • Onboard: GitHub repo, Google Colab, and **Google AI Studio** free Gemini API key; make one successful Gemini call.<br>• Exploratory data analysis on MedAESQA, MTSamples, PLABA, MedQuAD, and a Synthea sample.<br>• Build a **vector retrieval index** (Gemini embeddings + ChromaDB) over the MedlinePlus lay-language glossary and authoritative lay-health guidelines.<br>• Define the task spec, success rubric, and shared evaluation harness: Flesch-Kincaid, SMOG, medical-jargon density, output length, refusal rate.<br>• Build a **single-Gemini baseline (no agents)** on ~50 hand-curated examples — the comparison point for the multi-agent system.<br>• Each fellow chooses one of the five agents to own end-to-end. |
| October | Multi-Agent Pipeline (ADK) & Verifier Calibration | • Build the pipeline with **Google ADK**: a `SequentialAgent` chaining five `LlmAgent`s (Extractor → Simplifier → Verifier → Refiner → Readability).<br>• **Extractor**: structured-output prompting returning clinical atoms with types.<br>• **Simplifier**: RAG over the MedlinePlus / guidelines index.<br>• **Verifier**: LLM-as-judge scoring faithfulness, omission, addition, and reading-level match.<br>• **Refiner ↔ Verifier**: a **bounded** `LoopAgent(max_iterations=2)` — no open-ended recursion.<br>• Calibrate the Verifier against **MedAESQA's human accuracy / evidence-support labels**; target ≥80% agreement (Cohen's κ ≥ 0.6).<br>• First end-to-end multi-agent vs. baseline comparison across Gemini + Gemma + one external open model. |
| November | Ablations, Error Analysis & Leaderboard | • **Ablation studies**: drop one agent at a time and measure impact on each metric — this is the research contribution.<br>• **Error analysis**: which content types (lab values, drug names, procedures) produce the most hallucinations? Which agent catches them?<br>• **Leaderboard**: faithfulness, readability, latency, cost, and openness across all models.<br>• Begin the interactive demo (per-agent reasoning traces + evidence attributions). |
| December | Demo, Report & Presentation | • Finish the **app demo** on Hugging Face Spaces (free) or Cloud Run, showing per-agent traces, faithfulness scores, and flagged passages.<br>• Final deliverables: reproducible Colab notebooks, technical report, FaithfulMed leaderboard, ablation table, and team presentation.<br>• Optional: arXiv paper draft. |

> **Note for the team:** Create a GitHub Projects board in this repo to break these milestones into weekly tasks: **Projects** tab → **New project** → **Board** → a column per month.

---

## 📊 Datasets
All datasets are public and either de-identified, synthetic, or expert-curated. **No PHI/PII is involved.** You will not need every dataset for every agent — start with **MedAESQA** (evaluation backbone) plus **MTSamples + PLABA** (the simplification core). See `data/README.md` for exact download steps and licenses.

| Dataset | What it is | Role here | Format / Size | Where |
|---|---|---|---|---|
| **MedAESQA** ⭐ | 40 health questions × (1 expert + 30 machine answers); 7,651 evidence excerpts; **human accuracy & evidence-support judgments**; expert "nuggets" | **Primary faithfulness eval + Verifier calibration + Extractor gold** | `medaesqa_v1.json`, small (CC BY 4.0) | OSF: https://osf.io/ydbzq · code: https://github.com/deepaknlp/MedAESQA |
| **MTSamples** | ~5,000 de-identified transcribed medical reports | **Source documents** to simplify | text, <50 MB | https://www.mtsamples.com/ (also on Kaggle) |
| **PLABA** | Professional ↔ plain-language biomedical text pairs | **Gold pairs** for the Simplifier | JSON/text | https://osf.io/rnpmf/ (Attal et al., *Scientific Data* 2023) |
| **MedQuAD** | ~47k consumer-health Q&A (NIH/NLM) | Consumer-tone eval; RAG grounding | XML/JSON | https://github.com/abachaa/MedQuAD |
| **Synthea** | Fully synthetic patient records | **FHIR** stress test: discharge summaries / care plans | FHIR JSON, generate as needed | https://synthetichealth.github.io/synthea/ |
| **MedlinePlus glossary** | NLM lay-language definitions of medical terms | **RAG knowledge base** for the Simplifier | text/XML | https://medlineplus.gov/ |
| **MedQA / PubMedQA** *(optional)* | Standard medical-QA benchmarks (USMLE-style MCQ / abstract yes-no-maybe) | Optional extra measure of **clinical accuracy** | JSON | github.com/jind11/MedQA · pubmedqa.github.io |

⭐ **Start with MedAESQA.** It is the backbone of your evaluation: *expert human faithfulness labels* — exactly what you need to calibrate the Verifier (Cohen's κ target) — plus expert "nuggets" that serve as gold for the Extractor.

> **Scope note:** MedQA / PubMedQA test *clinical reasoning*, a different task shape than simplification/faithfulness. Treat them as **optional** supplementary benchmarks — the SME review flagged the 12-week cap, so don't let them pull focus from the core.

**Key details & gotchas**
- **MedAESQA is for *evaluation*, not training** — 40 questions is a rich gold set (it exceeds the 100-example calibration target) but too small to fine-tune on. Its 30 machine answers (documented in `data/MedAESQA_methods_M1-M30.xlsx`) give a built-in good→bad quality spread to test whether your Verifier separates faithful from unfaithful answers.
- **FHIR is nested and verbose** — write a flattener in September; never feed raw FHIR to the LLM.
- **MTSamples/Synthea aren't pre-paired for our task** — your ~50-example September baseline set is something you *build*; MedAESQA is the one you *download*.
- **Freeze the gold set early** — version MedAESQA + your curated examples and stop editing them. Silently changing your eval set is the #1 way to make all before/after numbers meaningless. *(This was explicit SME guidance.)*
- **Licensing:** MedAESQA is CC BY 4.0 (attribute the paper). PLABA is research-use — cite Attal et al. 2023, do not redistribute. MTSamples is de-identified; handle with care, don't re-post raw records.

---

## 🛠️ Suggested Approach

**ML Problem Type:** LLM / RAG / multi-agent orchestration (conditional text generation + LLM-as-judge evaluation). This is **not** classic supervised learning — there is no single loss to minimize. Your "model" is a *pipeline*; your "training" is prompt design, retrieval design, and calibration against human labels.

**Reference architecture (advisor guidance):**
Extractor (atoms) → Simplifier (+RAG) → Verifier (LLM-judge) → Refiner → Readability, with a **bounded** Refiner↔Verifier loop.

Two decisions that will save the team weeks — please take these seriously (they come directly from the SME technical review):
1. **Use a deterministic DAG, not free-form recursive agent loops.** Open-ended "agents call agents until satisfied" designs are hard to debug, blow through rate limits, and produce non-reproducible traces. In **Google ADK** this is a `SequentialAgent` for the main flow plus a `LoopAgent(max_iterations=2)` for the one place you want iteration (Refiner↔Verifier). A plain ordered Python function is a fine v1 too.
2. **Build the evaluation harness in Week 1–2, before the agents.** You can't improve what you can't measure. A starter lives in `notebooks/eval_harness.py`, and an ADK pipeline skeleton in `notebooks/adk_pipeline.py`.

**Recommended stack (GCP-first):**
- **Models:** **Gemini** (via Google AI Studio — free API key, no billing) as primary; **Gemma-2** open weights (runnable on the free Colab tier) for the open-model comparison; **one external open model** (e.g., Llama-3 via Groq, called through ADK's `LiteLlm`) to keep the leaderboard honest.
- **Orchestration:** **`google-adk`** (Agent Development Kit) — `SequentialAgent`, `LoopAgent`, `LlmAgent`. Model-agnostic, so one harness runs every model. *(LangGraph or CrewAI are acceptable alternatives.)*
- **RAG:** **Gemini `text-embedding-004`** + **ChromaDB** (offline fallback: `sentence-transformers` + `faiss-cpu`).
- **Readability & agreement:** `textstat` (Flesch-Kincaid, SMOG), `scikit-learn` (Cohen's κ).
- **Data / viz:** `pandas`, `numpy`, `openpyxl`, `matplotlib`, `seaborn`, `datasets`.
- **Demo:** `gradio` or `streamlit` on Hugging Face Spaces (free); Cloud Run is the GCP-native alternative (needs billing). ADK's built-in dev UI (`adk web`) is great for showing agent traces during development.

**Evaluation metrics:**
- Readability: Flesch-Kincaid Grade Level (primary), SMOG (secondary), medical-jargon density, output length, refusal rate.
- Faithfulness: the Verifier's score **calibrated against human labels** — report agreement (accuracy and Cohen's κ) vs. MedAESQA's human annotators; track omission and addition (hallucination) separately.
- System tradeoffs: latency, cost, and openness per model for the leaderboard.
- Golden rule: every headline number is reported on the *frozen* gold-standard set.

**Suggested per-fellow ownership** (one agent each, end-to-end):

| Agent | Owns | Core skill built |
|---|---|---|
| Extractor | Structured extraction of clinical "atoms" (facts + types) | Structured-output prompting, schema design |
| Simplifier | Plain-language rewriting with RAG grounding | RAG, retrieval design, prompt engineering |
| Verifier | LLM-as-judge faithfulness scoring + human calibration | Evaluation, annotation, inter-rater agreement |
| Refiner | Targeted fixes to flagged passages, bounded loop | Controlled generation, loop design |
| Readability | Grade-level targeting and tone control | Readability metrics, iterative rewriting |

---

## 📚 Resources to Get Started
**Problem space:** AHRQ and CDC "health literacy" resources (why ≤8th-grade is the target); studies on discharge-instruction readability.

**Google stack (start here):**
- Google AI Studio (free Gemini key) — https://aistudio.google.com
- Gemini API docs & `google-genai` SDK — https://ai.google.dev
- **Agent Development Kit (ADK)** docs & samples — https://google.github.io/adk-docs
- Gemma open models — https://ai.google.dev/gemma
- Vertex AI *(stretch; needs billing)* — https://cloud.google.com/vertex-ai

**Other docs & tutorials:**
- `textstat` readability library — https://pypi.org/project/textstat/
- ChromaDB (RAG) — https://docs.trychroma.com/
- Hugging Face Spaces + Gradio (demo) — https://huggingface.co/docs/hub/spaces

**Concepts to search:** LLM-as-a-judge (and its position/verbosity/self-preference biases); faithfulness / hallucination evaluation in summarization (FaithBench as a touchstone); Cohen's κ / inter-annotator agreement; FHIR basics (Condition, MedicationRequest, Procedure, CarePlan).

**Starter code (this repo):** `notebooks/eval_harness.py`, `notebooks/adk_pipeline.py`, and `requirements.txt`.

*Explore beyond these, and share anything interesting you find with the team!*

---

## 🤝 How We'll Work Together
**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of each month).

**Other ways to reach me:**
- Your team's channel in the Break Through Tech Discord (fastest for quick questions).
- Email: **sarita.ritu@gmail.com** — please cc your teammates and your AI Studio Coach.
- Ad-hoc Zoom / Google Meet check-ins by request when the team is blocked.
- *I aim to respond within 48 hours. For urgent, time-sensitive blockers, reach out to your AI Studio Coach first.*

**Recommended free tools:** GitHub (+ Projects board), Google Colab, Google AI Studio (Gemini) + Gemma, Hugging Face Spaces, Discord.

**Working norms:**
- Every experiment result gets committed (notebook + a row in the results table) — nothing lives only on one laptop.
- Keep a running decisions log so the December technical report writes itself.
- Rate limits are shared — coordinate large evaluation runs.

---

## 🚀 Getting Started
1. Read this overview and note questions for our first meeting.
2. Clone the repo, install `requirements.txt` in Colab, get a free **Google AI Studio** key, and make one successful Gemini call.
3. Download MedAESQA (see `data/README.md`); skim MTSamples and generate a few Synthea patients.
4. Run `notebooks/eval_harness.py` and skim `notebooks/adk_pipeline.py` to see the pipeline shape.
5. Read the GitHub Projects docs (https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects) and help set up the team board.
6. Claim an agent — Extractor / Simplifier / Verifier / Refiner / Readability.

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of **August 24th** (Break Through Tech's Bridge to Studio — Session C).
