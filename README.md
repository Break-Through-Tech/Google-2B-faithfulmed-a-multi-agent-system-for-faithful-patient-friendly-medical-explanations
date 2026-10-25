# FaithfulMed

FaithfulMed is a developing Google ADK multi-agent system for turning medical text into faithful,
patient-friendly explanations. Each agent lives in `agents/`; `notebooks/adk_pipeline.py` is the
current integration point while the team experiments with the workflow.

See [the current architecture guide](docs/CURRENT_ARCHITECTURE.md) for agent ownership, independent
agent usage, Verifier evaluation, dataset limits, and safe commands.

Offline unit tests (no API key and no Gemini call):

```bash
python3 -m unittest discover -s tests -v
```

Running any `run_*` function under `agents/` uses Google ADK, requires `GOOGLE_API_KEY`, and may use
Gemini API quota. Building the objects in `notebooks/adk_pipeline.py` does not itself call Gemini.

## Local agent workbench

After installing requirements in an activated virtual environment, start the local Streamlit workbench:

```bash
python -m streamlit run workbench.py
```

It opens a Verifier-only MedAESQA workbench. It reads `GOOGLE_API_KEY` from `.env`, shows the exact
inputs and raw output for every selected answer, compares predictions with human labels only after
generation, and caches completed results locally. See [the workbench guide](docs/VERIFIER_WORKBENCH.md).

The original team-project template follows below and can be updated as the broader project develops.

## Complete agent-system demo

The full-system Streamlit demo is separate from the individual-agent evaluation workbench. It accepts a pasted report or selected examples from the scratch, dev, and test MTSamples/PLABA splits, then runs the actual Extractor, Simplifier, production Verifier/Refiner loop, and Readability agents. One separate factual evaluator checks the final Readability output against the source. The page streams ADK model events and offers stage traces as JSON downloads.

From the repository root, with the virtual environment activated and `GOOGLE_API_KEY` saved in `.env`, run:

```bash
python -m streamlit run agentic_system_demo.py
```

The production loop permits at most two Verifier-to-Refiner revision cycles. Readability runs only after the production Verifier accepts the draft. If the second revision is still rejected, the latest draft remains available for inspection but is not labeled as a verified final explanation.

Choose MTSamples, PLABA, or both; choose `scratch`, `dev`, or `test`; and select only the examples you want to run. For de-identified dataset examples, **Run selected examples** reuses an append-only cached record only when the dataset identity, source and reference content, model settings, prompt hashes, and evaluator/implementation versions all match. **Force new run** bypasses the cache and appends a new uniquely identified record without deleting prior runs or inheriting their manual labels. These records live under the git-ignored `eval_results/` directory. Open **Saved run history** to restore an older result and continue its review. Pasted reports are kept only in Streamlit session state unless you explicitly download their JSON trace.

In **Human claim review**, every factual-evaluator claim labeled `unsupported` or `contradicted` and `clinically_meaningful: true` is shown with its exact saved claim text, source spans, and evaluator reason. It defaults to **Hallucination**, matching Gemini's evaluator verdict, until you select **Hallucination**, **Not hallucination**, or **Ignore**. Decisions are appended separately to `eval_results/system_claim_reviews.jsonl` under the immutable `run_id` and `claim_id`, so widget reruns, browser refreshes, cached runs, and history reopening preserve them without another model call. Ignored claims are excluded from claim-level calculations. The adjusted rates remain provisional whenever Gemini-default decisions have not been human-reviewed.

The automatic metrics panel remains separate and reports Flesch–Kincaid grade, SMOG, word count, refusal rate, the existing Dale–Chall difficult-word proxy (clearly labeled as a jargon-density proxy), and independent automated estimates of source fidelity and clinically meaningful hallucination rate. Readability is computed from the exact final displayed explanation, including bullet punctuation. The refusal detector is a phrase-based heuristic. Automated factual and screening labels are not human-reviewed gold labels, so challenge factuality targets are not declared met from screening alone. MTSamples adjudicated facts and PLABA adaptations are post-generation references only; they are never supplied to the generation pipeline. PLABA screening uses exact professional-source sentence IDs rather than assuming MTSamples-style fact annotations, and the plain-language adaptation is not treated as required wording.

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Taylor Nguyen    | @taylornguyen | Data exploration, visualization, overall project coordination            |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


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

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
