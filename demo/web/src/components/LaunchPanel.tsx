import { ClipboardPaste, Database, History, Lock, Play, RefreshCcw, Square, Zap } from "lucide-react";
import { useState } from "react";
import type { Config, LiveExample, SavedRun } from "../lib/types";
import { Chip, fmt } from "./ui";

type Tab = "replay" | "dataset" | "paste";

export type LaunchRequest =
  | { mode: "replay"; runId: string; speed: number }
  | { mode: "live"; exampleKey: string }
  | { mode: "live"; sourceText: string };

const SPEEDS = [0.5, 1, 2, 4];

export function LaunchPanel({
  config,
  runs,
  examples,
  busy,
  onLaunch,
  onStop,
}: {
  config: Config | null;
  runs: SavedRun[];
  examples: LiveExample[];
  busy: boolean;
  onLaunch: (request: LaunchRequest) => void;
  onStop: () => void;
}) {
  const [tab, setTab] = useState<Tab>("replay");
  const [runId, setRunId] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1);
  const [exampleKey, setExampleKey] = useState<string>("");
  const [pasted, setPasted] = useState("");
  const live = config?.live_available ?? false;
  const selectedRun = runId ?? runs[0]?.run_id ?? null;
  const selectedExample = examples.find((e) => e.key === exampleKey) ?? examples[0];

  const tabs: { key: Tab; label: string; icon: typeof History; locked: boolean }[] = [
    { key: "replay", label: "Recorded", icon: History, locked: false },
    { key: "dataset", label: "Live dataset", icon: Database, locked: !live },
    { key: "paste", label: "Live paste", icon: ClipboardPaste, locked: !live },
  ];

  const canRun =
    tab === "replay" ? Boolean(selectedRun) : tab === "dataset" ? Boolean(selectedExample) : pasted.trim().length > 40;

  const launch = () => {
    if (tab === "replay" && selectedRun) onLaunch({ mode: "replay", runId: selectedRun, speed });
    else if (tab === "dataset" && selectedExample) onLaunch({ mode: "live", exampleKey: selectedExample.key });
    else if (tab === "paste") onLaunch({ mode: "live", sourceText: pasted.trim() });
  };

  return (
    <div className="glass flex flex-col p-4">
      <div className="mb-3 grid grid-cols-3 gap-1 rounded-xl bg-ink-950/60 p-1">
        {tabs.map(({ key, label, icon: Icon, locked }) => (
          <button
            key={key}
            disabled={locked || busy}
            onClick={() => setTab(key)}
            title={locked ? "Live runs need a Gemini API key on the server" : undefined}
            className={`flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-xs font-medium transition ${
              tab === key ? "bg-ink-700 text-white shadow" : "text-slate-400 hover:text-slate-200"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            {locked ? <Lock className="h-3.5 w-3.5" /> : <Icon className="h-3.5 w-3.5" />}
            {label}
          </button>
        ))}
      </div>

      {tab === "replay" && (
        <>
          <p className="mb-2 text-xs text-slate-400">
            Real Gemini outputs recorded from earlier full-pipeline runs, replayed with compressed timing. No API calls.
          </p>
          <div className="scroll-thin -mx-1 max-h-[22rem] space-y-1.5 overflow-y-auto px-1">
            {runs.map((run) => (
              <button
                key={run.run_id}
                disabled={busy}
                onClick={() => setRunId(run.run_id)}
                className={`w-full rounded-xl border p-3 text-left transition ${
                  selectedRun === run.run_id ? "border-cyan-400/50 bg-cyan-400/[0.06]" : "border-white/[0.06] bg-ink-850 hover:border-white/15"
                }`}
              >
                <div className="mb-1 flex flex-wrap items-center gap-1.5">
                  <span className="text-sm font-semibold text-slate-100">{run.example_id}</span>
                  {run.revisions > 0 && (
                    <Chip className="bg-pink-400/10 text-pink-300 ring-pink-400/30">
                      <RefreshCcw className="h-3 w-3" /> self-corrects
                    </Chip>
                  )}
                  <span className="ml-auto text-[11px] text-slate-500">{run.dataset}</span>
                </div>
                <p className="line-clamp-2 text-xs text-slate-400">{run.preview}</p>
                <div className="mt-1.5 flex gap-3 text-[11px] text-slate-500">
                  <span>grade {fmt(run.fk_grade)}</span>
                  <span>fidelity {fmt(run.fidelity, 0, "%")}</span>
                </div>
              </button>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
            Speed
            <div className="flex gap-1 rounded-lg bg-ink-950/60 p-0.5">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`rounded-md px-2 py-1 font-medium ${speed === s ? "bg-ink-700 text-white" : "hover:text-slate-200"}`}
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {tab === "dataset" && (
        <>
          <p className="mb-2 text-xs text-slate-400">De-identified scratch/dev examples. Runs every agent live on {config?.model}; takes about 1–5 minutes.</p>
          <select
            value={selectedExample?.key ?? ""}
            onChange={(e) => setExampleKey(e.target.value)}
            disabled={busy}
            className="w-full rounded-xl border border-white/10 bg-ink-850 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/50"
          >
            {examples.map((e) => (
              <option key={e.key} value={e.key}>
                {e.dataset} · {e.split} · {e.example_id} ({Math.round(e.chars / 100) / 10}k chars)
              </option>
            ))}
          </select>
          <pre className="scroll-thin mt-2 max-h-72 overflow-y-auto whitespace-pre-wrap rounded-xl bg-ink-950/60 p-3 font-serif text-xs leading-relaxed text-slate-400">
            {selectedExample?.source_text}
          </pre>
        </>
      )}

      {tab === "paste" && (
        <>
          <p className="mb-2 text-xs text-amber-200/80">Use de-identified or synthetic text only. Nothing is saved on the server.</p>
          <textarea
            value={pasted}
            onChange={(e) => setPasted(e.target.value.slice(0, config?.max_source_chars ?? 15000))}
            disabled={busy}
            placeholder="Paste a clinical note, discharge summary, or abstract…"
            className="scroll-thin h-72 w-full resize-none rounded-xl border border-white/10 bg-ink-850 p-3 font-serif text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-400/50"
          />
          <div className="mt-1 text-right text-[11px] text-slate-500">
            {pasted.length.toLocaleString()} / {(config?.max_source_chars ?? 15000).toLocaleString()}
          </div>
        </>
      )}

      {busy ? (
        <button onClick={onStop} className="mt-4 flex items-center justify-center gap-2 rounded-xl border border-rose-400/40 bg-rose-500/10 py-3 text-sm font-semibold text-rose-200 hover:bg-rose-500/20">
          <Square className="h-4 w-4" /> Stop
        </button>
      ) : (
        <button
          onClick={launch}
          disabled={!canRun}
          className="group relative mt-4 flex items-center justify-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-cyan-400 via-sky-400 to-violet-400 py-3 text-sm font-bold text-ink-950 shadow-[0_0_30px_-8px_#22d3ee] transition hover:shadow-[0_0_40px_-6px_#22d3ee] disabled:opacity-40"
        >
          {tab === "replay" ? <Play className="h-4 w-4" /> : <Zap className="h-4 w-4" />}
          {tab === "replay" ? "Replay pipeline" : "Run agents live"}
        </button>
      )}
    </div>
  );
}
