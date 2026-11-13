# FaithfulMed demo: React frontend + FastAPI backend in one container.
# Hugging Face Spaces (Docker SDK) serves port 7860; Cloud Run sets $PORT.

FROM node:22-slim AS web
WORKDIR /web
COPY demo/web/package.json demo/web/package-lock.json ./
RUN npm ci
COPY demo/web/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PORT=7860
RUN useradd -m -u 1000 user
WORKDIR /app
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

COPY agents/ agents/
COPY evals/*.py evals/
COPY notebooks/eval_harness.py notebooks/
COPY demo/*.py demo/
COPY data/MTSamples/scratch/ data/MTSamples/scratch/
COPY data/MTSamples/dev/ data/MTSamples/dev/
COPY data/PLABA/scratch/ data/PLABA/scratch/
COPY data/PLABA/dev/ data/PLABA/dev/
COPY eval_results/system_demo_runs.jsonl eval_results/
COPY --from=web /web/dist demo/web/dist

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn demo.server:app --host 0.0.0.0 --port ${PORT}"]
