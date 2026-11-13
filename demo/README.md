# FaithfulMed interactive demo

A React + Tailwind frontend on a FastAPI backend that imports the real agent code in `agents/` and streams the pipeline to the browser as it runs. It shows:

- **Per-agent reasoning traces.** A live pipeline graph and a card for every agent call, with tokens streaming in as Gemini generates them. Finished cards switch to a structured view: extracted facts, the Verifier's verdict with the issues it sent to the Refiner, a word-level diff of exactly what the Refiner changed, and the final text.
- **Faithfulness scores.** Fact-check fidelity, Flesch–Kincaid grade (source → final), clinically meaningful flags, and how many refinement passes ran.
- **Evidence attribution.** Click any sentence of the final explanation to highlight the source sentences the independent fact-check linked it to, with the evaluator's reason.
- **Flagged passages.** Every claim the fact-check did not mark as supported, in order of severity.

## Modes

| Mode | What runs | Needs |
|---|---|---|
| **Recorded** | Replays a saved run from `eval_results/system_demo_runs.jsonl` with compressed timing. Real model outputs, no API calls. | nothing |
| **Live dataset** | Runs every agent on a de-identified scratch/dev MTSamples or PLABA example. Takes about 1–5 minutes. | `GOOGLE_API_KEY` |
| **Live paste** | Same, on pasted text (up to 15,000 characters). Nothing is saved on the server. | `GOOGLE_API_KEY` |

Recorded mode is the safe choice for presentations: it can't hit a Gemini outage or quota limit. `PLABA_SCRATCH_74` is listed first because the Verifier catches two unsupported claims and the Refiner removes them.

## Run locally

From the repository root, with `.venv` active and `GOOGLE_API_KEY` in `.env`:

```bash
# one-time: build the frontend
cd demo/web && npm install && npm run build && cd ../..

python -m uvicorn demo.server:app --port 8000
# open http://localhost:8000
```

Frontend development with hot reload (keeps the API on port 8000):

```bash
python -m uvicorn demo.server:app --reload --port 8000   # terminal 1
cd demo/web && npm run dev                                # terminal 2, open http://localhost:5173
```

Offline tests (fake Gemini, no quota): `python -m unittest discover -s tests -p "test_demo_server.py"`

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | none | Enables the live modes and the Gemini and Gemma models. Without it, only Recorded mode is available. |
| `GROQ_API_KEY` | none | Optional: enables gpt-oss-120b (via Groq) in the model picker. |
| `DEMO_LIVE` | `1` | Set to `0` to turn off live runs even when a key is present. |
| `DEMO_MAX_LIVE_RUNS` | `1` | Concurrent live runs allowed. Extra requests get HTTP 429. |
| `DEMO_MODEL` | `gemini-3.8-flash` | Fact-check model, and the picker's default when it matches a listed model. |
| `UMLS_API_KEY` | none | Optional: enables the Term Lookup stage. |

Live runs have a **model picker**. It lists the models in `evals/leaderboard_models.json` (the same list the leaderboard uses) and locks any whose API key isn't set on the server. Every agent runs on the chosen model; the fact-check stays on `DEMO_MODEL` so scores are comparable across models. Recorded runs show the model they were made with.

If the browser disconnects mid-run, the server cancels the pipeline so it stops using quota.

## Deploy to Hugging Face Spaces (free)

1. Create a Space with **SDK: Docker** (blank template).
2. Push this repository to the Space, or copy in at least `Dockerfile`, `.dockerignore`, `agents/`, `evals/`, `notebooks/eval_harness.py`, `demo/`, the scratch/dev folders of `data/MTSamples` and `data/PLABA`, and `eval_results/system_demo_runs.jsonl`.
3. The Space's `README.md` must start with this header:

   ```yaml
   ---
   title: FaithfulMed
   emoji: 🩺
   colorFrom: blue
   colorTo: purple
   sdk: docker
   app_port: 7860
   ---
   ```

4. Optional: to enable live runs, add `GOOGLE_API_KEY` under **Settings → Variables and secrets** as a **secret**. Never commit it. Leave it unset for a replay-only public demo.

## Deploy to Cloud Run

```bash
gcloud run deploy faithfulmed-demo --source . --region us-central1 --allow-unauthenticated \
  --set-secrets GOOGLE_API_KEY=<secret-name>:latest --timeout 900
```

`--timeout 900` keeps long live runs from being cut off. Cloud Run sets `$PORT` and the container uses it.

## Files

```
demo/
├── server.py        FastAPI app: /api/config, /api/runs, /api/examples, /api/replay/{id}, /api/run
├── events.py        Turns pipeline callbacks and saved records into one SSE event contract
├── requirements.txt Slim runtime dependencies for the container
└── web/             Vite + React + Tailwind frontend (build output in web/dist, git-ignored)
Dockerfile           Two-stage build: Node builds the frontend, Python serves everything
```
