# FaithfulMed

FaithfulMed is a project in partnership with Google. It sits at the intersection of two Google-relevant themes — trustworthy generative AI and equitable access to information — and is built on Google's own open stack (Gemini and the Agent Development Kit). FaithfulMed is a Google ADK multi-agent system for turning medical text into faithful, patient-friendly explanations. The project sits at the intersection of health literacy and trustworthy generative AI: it is designed to reduce patient-facing medical jargon while keeping the explanation grounded in the original source material rather than drifting into unsupported or invented claims.

The repository is structured around a small set of specialized agents in `agents/`, an end-to-end pipeline in `notebooks/adk_pipeline.py`, and a runtime demo in `agentic_system_demo.py`.

## 🎯 Project Highlights

- Developed a Gogogle ADK multi-agent LLM system - Extractor, Simplifier, Verifier, Refiner, and Readability agents - to turn clinical reports into patient-friendly explanations while preserving faithfulness. 
- Achieved ≥80% of outputs at ≤8th-grade Flesch-Kincaid reading level, ≥85% factual fidelity on the human-annotated test set, and <10% of outputs contain a clinically meaningful unsupported claim.
- Following two Google-relevant themes - trustworthy generative AI and equitable access to information.
- System is intended to improve readability without sacrificing source fidelity.
- Designed to support comparison against a single-prompt baseline and ablation studies.

## 👩🏽‍💻 Setup and Installation

### 1. Clone the Repository

First, you must clone the repository and move into the directory:

```bash
git clone https://github.com/Break-Through-Tech/Google-2B-faithfulmed-a-multi-agent-system-for-faithful-patient-friendly-medical-explanations.git
cd Google-2B-faithfulmed-a-multi-agent-system-for-faithful-patient-friendly-medical-explanations
```

### 2. Virtual Environment and Dependencies

Next, you must create the virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure Gemini API Access

In order to run the agents and/or the pipeline, you must have a Gemini API key. To do this, create a Gemini API key here: [click me!](https://www.google.com/aclk?sa=L&pf=1&ai=DChsSEwjW3pDv_fuXAxUawp8JHYxuLl8YACICCAEQARoCbWQ&co=1&ase=2&gclid=Cj0KCQiA-8XXBhDaARIsADnEOrBy23NwmUKe06wIU53VJjxtL_3xEXNAOm0DNCPPV2LyvXEP8nlooZMaAh-cEALw_wcB&ei=hU_yapDlDc-IptQPuLKJ6QQ&cid=CAAS0gHkaKTAdw8st9BLEnLvZ4UoTK5J3P67QQLo7jY7KzhYvD2q2arW-NdJ2LKSKBMJ7F_EF4d4uQtSR3oFjkdlzKF6LJrsPH7kMFQa8TK2ipSZj29gWefzXwnIOkTgm5Z52PH9sZOXflHADphpUXPPcGLH2nedrsBAoaVpSJoXVQ2Si3VKRBpPu1SIMJpQo3ACoSLFdWZX9vF4AFNJ0Nb7FAc4_TQC09E8410huQjLD4d6FYpVzJpNCdZGq4IQIKpnF7HBBjb6QcK5hw1gOQ3rWmDS-_c&cce=2&category=acrcp_v1_32&sig=AOD64_3P9MPYZ24lPhcQtOc3Qn1feV5tJA&q&sqi=2&nis=4&adurl=https://aistudio.google.com/docs/api-key?utm_source%3Dgoogle%26utm_medium%3Dcpc%26utm_campaign%3DCloud-SS-DR-AIS-FY26-global-gsem-1713578%26utm_content%3Dtext-ad%26utm_term%3DKW_google%2520api%2520key%26gad_source%3D1%26gad_campaignid%3D23417416052%26gbraid%3D0AAAAACn9t66hx3Xz4iXoeTSGt8F9NDREV%26gclid%3DCj0KCQiA-8XXBhDaARIsADnEOrBy23NwmUKe06wIU53VJjxtL_3xEXNAOm0DNCPPV2LyvXEP8nlooZMaAh-cEALw_wcB&ved=2ahUKEwjQu4rv_fuXAxVPhIkEHThZIk0QqyQoAHoECCgQIQ)


After getting your Gemini API key, run this command:

```bash
cp .env.example .env
```

Once the .env file is created, paste your key into the GOOGLE_API_KEY variable.

### 4. Run Tests, Workbench, and Interactive Demo

#### Tests

Run the offline unit tests without an API key:

```bash
python3 -m unittest discover -s tests -v
```

#### Interactive Demo

To launch the end-to-end demo, with the virtual environment active and .env configured:

```bash
python -m streamlit run agentic_system_demo.py
```

#### Local Agent Workbench

After installing requirements in an activated virtual environment, start the local Streamlit workbench:

```bash
python -m streamlit run workbench.py
```

The Streamlit workbench provides separate views for the Baseline, Verifier, Refiner, and Readability agents, selected from the sidebar. The Baseline view reports evaluation metrics; the Verifier view evaluates MedAESQA answers against human labels after generation; and the Refiner and Readability views support agent-specific testing.

## Complete Agent-System Demo

The full-system Streamlit demo is separate from the individual-agent evaluation workbench. It accepts a pasted report or selected examples from the scratch, dev, and test MTSamples/PLABA splits, then runs the actual Extractor, Simplifier, production Verifier/Refiner loop, and Readability agents. One separate factual evaluator checks the final Readability output against the source. The page streams ADK model events and offers stage traces as JSON downloads.

From the repository root, with the virtual environment activated and `GOOGLE_API_KEY` saved in `.env`, run:

```bash
python -m streamlit run agentic_system_demo.py
```

The production loop permits at most two Verifier-to-Refiner revision cycles. Readability runs only after the production Verifier accepts the draft. If the second revision is still rejected, the latest draft remains available for inspection but is not labeled as a verified final explanation.

Choose MTSamples, PLABA, or both; choose `scratch`, `dev`, or `test`; and select only the examples you want to run. For de-identified dataset examples, **Run selected examples** reuses an append-only cached record only when the dataset identity, source and reference content, model settings, prompt hashes, and evaluator/implementation versions all match. **Force new run** bypasses the cache and appends a new uniquely identified record without deleting prior runs or inheriting their manual labels. These records live under the git-ignored `eval_results/` directory. Open **Saved run history** to restore an older result and continue its review. Pasted reports are kept only in Streamlit session state unless you explicitly download their JSON trace.

In **Human claim review**, every factual-evaluator claim labeled `unsupported` or `contradicted` and `clinically_meaningful: true` is shown with its exact saved claim text, source spans, and evaluator reason. It defaults to **Hallucination**, matching Gemini's evaluator verdict, until you select **Hallucination**, **Not hallucination**, or **Ignore**. Decisions are appended separately to `eval_results/system_claim_reviews.jsonl` under the immutable `run_id` and `claim_id`, so widget reruns, browser refreshes, cached runs, and history reopening preserve them without another model call. Ignored claims are excluded from claim-level calculations. The adjusted rates remain provisional whenever Gemini-default decisions have not been human-reviewed.

The automatic metrics panel remains separate and reports Flesch–Kincaid grade, SMOG, word count, refusal rate, the existing Dale–Chall difficult-word proxy (clearly labeled as a jargon-density proxy), and independent automated estimates of source fidelity and clinically meaningful hallucination rate. Readability is computed from the exact final displayed explanation, including bullet punctuation. The refusal detector is a phrase-based heuristic. Automated factual and screening labels are not human-reviewed gold labels, so challenge factuality targets are not declared met from screening alone. MTSamples adjudicated facts and PLABA adaptations are post-generation references only; they are never supplied to the generation pipeline. PLABA screening uses exact professional-source sentence IDs rather than assuming MTSamples-style fact annotations, and the plain-language adaptation is not treated as required wording.

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Nirmit Bhoyar   | @Nirmit651 | Data preprocessing, Verifier agent, ablation study, overall project coordination |
| Brenda Romero   | @brendaromero3 | Data preprocessing, Readability agent, Baseline agent  |
| Supriyaa Hejib  | @supriyaahejib | Data preprocessing, Simplifier agent, error analysis |
| Tabitha Githinji | @GithinjiTabz1 | Data preprocessing, Extractor agent, leaderboard |
| Riya Patel      | @riyampate     | Data preprocessing, Simplifier agent |
| Sameen Chand    | @sameenchand   | Data preprocessing, Refiner agent, interactive demo |

---

## 🏗️ **Project Overview**

FaithfulMed is a Break Through Tech AI Studio project in partnership with Google. FaithfulMed aligns with the challenge goal of building a trustworthy clinical explanation system for patient-friendly communication. The core challenge is not only readability, but source fidelity: the final explanation must remain faithful to the original clinical source rather than sounding plausible but unsupported.

The project uses clinical text and a multi-agent LLM workflow to produce explanations that are both easy to read and faithful to the source. The main research question is whether specialized agents produce better explanations than a single well-prompted model, and which agents matter for each goal.

---

## 📊 **Data Exploration**

The project relies on public clinical and biomedical datasets documented in `data/README.md` and `Challenge-Project-Overview.md`.

### Primary evaluation data
- MedAESQA is the primary evaluation backbone.
- It contains 40 health questions, with human accuracy and evidence-support judgments, and 7,651 evidence excerpts.
- It is intended for Verifier calibration and faithfulness evaluation, not for training.

### Source corpora used for simplification
- MTSamples: de-identified transcribed medical reports used as source documents to simplify.
    - 42 MTSamples reports split across dev, scratch, and test sets.
    - Individually annotated and adjudicated between pairs. 
- PLABA: professional-to-plain-language pairs intended as a gold-style simplification resource.
    - 28 PLABA reports split across dev, scratch, and test sets.

---

## 🧠 **Model Development**

The project is implemented as a multi-agent LLM workflow rather than a single model. The system uses Google's gemini-3.8-flash.

### Agent roles in the current code
- Extractor: identifies clinically meaningful facts from a medical source using a structured output schema defined in `agents/extractor.py`.
- Simplifier: rewrites medical text into patient-friendly language while preserving meaning; its instruction is in `agents/simplifier.py`.
- Verifier: checks whether the explanation is faithful to the original report; its instruction is in `agents/verifier.py`.
- Refiner: makes targeted revisions based on Verifier feedback; its instruction is in `agents/refiner.py`.
- Readability: rewrites the accepted draft to improve readability while preserving clinical meaning; its instruction is in `agents/readability.py`.

### Pipeline structure
The experimental pipeline is assembled in `notebooks/adk_pipeline.py` and the production-style orchestration is implemented in `agents/production_pipeline.py`.

The flow is:

1. Extractor reads the source report and produces structured facts.
2. Simplifier turns the source and facts into a patient-friendly draft.
3. Verifier checks the draft against the source.
4. Refiner revises the draft when the Verifier identifies unsupported claims or omissions.
5. Readability runs after acceptance to improve readability.

---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements**

We are deeply grateful to our Challenge Advisor, Sarita, for her continued guidance and encouragement throughout the project. Her thoughtful answers to our endless questions helped us build a strong foundation and better understand the work ahead. By sharing her personal perspective on the importance of understanding one's medical reports and health history, she inspired us to approach this project with care and dedication. 

We're also grateful to our Coach, Rohil, for his support and thoughtful check-ins throughout the project. His encouragement helped us stay grounded, work through challenges, and keep moving forward as a team. He also encouraged open communication, helping us stay connected and aligned as a team.

We appreciate the time and care Sarita and Rohil invested in supporting our progress. We couldn't have done it without them!

---