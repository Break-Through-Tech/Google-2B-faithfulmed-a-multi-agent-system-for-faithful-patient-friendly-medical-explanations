import { AlertOctagon, Github, Radio, Stethoscope, Timer } from "lucide-react";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { EvidenceView, type Selection } from "./components/EvidenceView";
import { FlaggedPassages } from "./components/FlaggedPassages";
import { LaunchPanel, type LaunchRequest } from "./components/LaunchPanel";
import { Markdown } from "./components/Markdown";
import { PipelineGraph } from "./components/PipelineGraph";
import { ScoreBoard } from "./components/ScoreBoard";
import { Timeline } from "./components/Timeline";
import { SectionTitle } from "./components/ui";
import { AGENTS } from "./lib/agents";
import { initialRunState, runReducer } from "./lib/runState";
import { streamEvents } from "./lib/stream";
import { friendlyError } from "./lib/text";
import type { Config, LiveExample, SavedRun } from "./lib/types";

const REPO = "https://github.com/Break-Through-Tech/Google-2B-faithfulmed-a-multi-agent-system-for-faithful-patient-friendly-medical-explanations";

function Elapsed({ start, end }: { start: number | null; end: number | null }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!start || end) return;
    const id = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(id);
  }, [start, end]);
  if (!start) return null;
  return <span className="tabular-nums">{(((end ?? now) - start) / 1000).toFixed(1)}s</span>;
}

function Hero() {
  const order = ["Extractor", "TermLookup", "Simplifier", "Verifier", "Refiner", "Readability", "Evaluator"];
  return (
    <div className="glass grid-bg flex h-full flex-col justify-center p-8">
      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-300/80">Google ADK · Gemini multi-agent system</div>
      <h1 className="mt-3 max-w-2xl text-3xl font-extrabold leading-tight tracking-tight text-slate-50 sm:text-4xl">
        Medical jargon in. <span className="text-gradient">A faithful explanation out.</span>
      </h1>
      <p className="mt-4 max-w-2xl text-slate-400">
        Seven specialized agents read a clinical report, explain it at an 8th-grade level, and check every sentence against the source. Pick a run on the left and watch them work.
      </p>
      <div className="mt-8 grid gap-2 sm:grid-cols-2">
        {order.map((key, i) => {
          const a = AGENTS[key];
          return (
            <div key={key} className="flex items-center gap-3 rounded-xl border border-white/[0.06] bg-ink-900/60 p-3">
              <span className="font-mono text-[10px] text-slate-600">{String(i + 1).padStart(2, "0")}</span>
              <a.icon className={`h-5 w-5 shrink-0 ${a.text}`} />
              <div className="min-w-0">
                <div className="text-sm font-semibold text-slate-100">{a.label}</div>
                <div className="truncate text-xs text-slate-500">{a.role}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [runs, setRuns] = useState<SavedRun[]>([]);
  const [examples, setExamples] = useState<LiveExample[]>([]);
  const [run, dispatch] = useReducer(runReducer, initialRunState);
  const [selection, setSelection] = useState<Selection>(null);
  const [follow, setFollow] = useState(true);
  const abort = useRef<AbortController | null>(null);
  const evidenceRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/config").then((r) => r.json()).then(setConfig).catch(() => undefined);
    fetch("/api/runs").then((r) => r.json()).then(setRuns).catch(() => undefined);
    fetch("/api/examples").then((r) => r.json()).then(setExamples).catch(() => undefined);
  }, []);

  const launch = useCallback(async (request: LaunchRequest) => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setSelection(null);
    dispatch({ type: "begin" });
    window.scrollTo({ top: 0, behavior: "smooth" });
    const [url, init]: [string, RequestInit] =
      request.mode === "replay"
        ? [`/api/replay/${request.runId}?speed=${request.speed}`, {}]
        : [
            "/api/run",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify("exampleKey" in request ? { example_key: request.exampleKey } : { source_text: request.sourceText }),
            },
          ];
    try {
      await streamEvents(url, init, dispatch, controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) dispatch({ type: "client_error", message: (error as Error).message });
    }
  }, []);

  const stop = () => {
    abort.current?.abort();
    dispatch({ type: "client_error", message: "Stopped." });
  };

  const selectAndScroll = (s: Selection) => {
    setSelection(s);
    evidenceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const busy = run.phase === "running" || run.phase === "evaluating";
  const meta = run.meta;

  return (
    <div className="mx-auto max-w-[1400px] px-4 pb-24 sm:px-6">
      <header className="flex flex-wrap items-center gap-3 py-5">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 shadow-[0_0_24px_-4px_#22d3ee]">
          <Stethoscope className="h-5 w-5 text-ink-950" />
        </div>
        <div>
          <div className="text-lg font-bold tracking-tight text-slate-50">FaithfulMed</div>
          <div className="text-xs text-slate-500">Faithful, patient-friendly medical explanations</div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2 text-xs">
          {meta && (
            <span
              className={`flex items-center gap-1.5 rounded-full px-3 py-1 font-semibold ring-1 ring-inset ${
                meta.mode === "live" ? "bg-rose-500/10 text-rose-300 ring-rose-500/30" : "bg-sky-400/10 text-sky-300 ring-sky-400/30"
              }`}
            >
              <Radio className={`h-3.5 w-3.5 ${busy && meta.mode === "live" ? "animate-pulse" : ""}`} />
              {meta.mode === "live" ? "LIVE" : `REPLAY · recorded ${meta.recorded_at_utc?.slice(0, 10)}`}
            </span>
          )}
          {run.startedAt && (
            <span className="flex items-center gap-1.5 rounded-full bg-white/5 px-3 py-1 text-slate-300 ring-1 ring-inset ring-white/10">
              <Timer className="h-3.5 w-3.5" /> <Elapsed start={run.startedAt} end={run.finishedAt} />
            </span>
          )}
          <span className="rounded-full bg-white/5 px-3 py-1 font-mono text-slate-400 ring-1 ring-inset ring-white/10">{meta?.model ?? config?.model ?? "gemini"}</span>
          <a href={REPO} target="_blank" rel="noreferrer" className="rounded-full bg-white/5 p-1.5 text-slate-400 ring-1 ring-inset ring-white/10 hover:text-white" aria-label="GitHub repository">
            <Github className="h-4 w-4" />
          </a>
        </div>
      </header>

      <div className="grid gap-5 lg:grid-cols-12">
        <aside className="space-y-4 lg:col-span-4 lg:self-start xl:col-span-3 lg:sticky lg:top-4">
          <LaunchPanel config={config} runs={runs} examples={examples} busy={busy} onLaunch={launch} onStop={stop} />
          {meta && (
            <details className="glass group p-4" open={false}>
              <summary className="cursor-pointer select-none text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                Input report · {meta.example_id}
              </summary>
              <pre className="scroll-thin mt-3 max-h-80 overflow-y-auto whitespace-pre-wrap font-serif text-xs leading-relaxed text-slate-400">{meta.source_text}</pre>
            </details>
          )}
        </aside>

        <main className="min-w-0 space-y-5 lg:col-span-8 xl:col-span-9">
          {run.phase === "idle" ? (
            <Hero />
          ) : (
            <>
              <PipelineGraph run={run} maxRevisions={config?.max_revision_cycles ?? 2} />
              {run.error && (
                <div className="glass flex items-center gap-3 border-rose-500/30 p-4 text-sm text-rose-200">
                  <AlertOctagon className="h-5 w-5 shrink-0" /> {friendlyError(run.error)}
                </div>
              )}
              <section>
                <SectionTitle
                  kicker="Per-agent reasoning trace"
                  title="Watch each agent work"
                  right={
                    <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400">
                      <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} className="accent-cyan-400" />
                      Follow active agent
                    </label>
                  }
                />
                <Timeline run={run} follow={follow} />
              </section>
              {run.pipeline && !run.pipeline.accepted && (
                <div className="glass border-amber-400/30 p-5">
                  <div className="mb-2 font-semibold text-amber-200">
                    {run.pipeline.status === "not_accepted" ? "The Verifier never accepted a draft, so no explanation is released." : "The pipeline stopped."}
                  </div>
                  {run.pipeline.error && <p className="text-sm text-slate-400">{friendlyError(run.pipeline.error)}</p>}
                  {run.pipeline.latestDraft && <Markdown text={run.pipeline.latestDraft} className="mt-3 text-sm text-slate-400" />}
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {run.result && (
        <div className="animate-fade-up mt-10 space-y-8">
          <section>
            <SectionTitle kicker="Faithfulness scores" title="How the final explanation measures up" />
            <ScoreBoard result={run.result} run={run} />
          </section>
          {run.result.explanation_spans.length > 0 && (
            <section ref={evidenceRef} className="scroll-mt-4">
              <SectionTitle kicker="Evidence attribution" title="Every sentence, traced to its source" />
              <EvidenceView result={run.result} selection={selection} onSelect={setSelection} />
            </section>
          )}
          <section>
            <SectionTitle kicker="Flagged passages" title="What the fact-check wants a human to look at" />
            <FlaggedPassages result={run.result} onSelect={selectAndScroll} />
          </section>
          <p className="text-center text-xs text-slate-600">
            Scores come from an automated, source-based evaluator and are estimates, not human gold labels. Not medical advice.
          </p>
        </div>
      )}
    </div>
  );
}
