import { Check, Loader2, Minus, RefreshCcw, X } from "lucide-react";
import { agentInfo } from "../lib/agents";
import type { RunState } from "../lib/runState";
import type { Stage } from "../lib/types";

type NodeStatus = "waiting" | "running" | "completed" | "failed" | "skipped";

function statusOf(stages: Stage[]): NodeStatus {
  if (!stages.length) return "waiting";
  const last = stages[stages.length - 1];
  return last.status as NodeStatus;
}

function Node({ agent, status, count, compact }: { agent: string; status: NodeStatus; count?: number; compact?: boolean }) {
  const info = agentInfo(agent);
  const Icon = info.icon;
  const running = status === "running";
  const done = status === "completed";
  return (
    <div className={`flex flex-col items-center gap-2 text-center ${compact ? "w-24" : "w-24 sm:w-28"}`}>
      <div
        className={`relative grid h-14 w-14 place-items-center rounded-2xl border transition-all duration-500 ${
          running
            ? "animate-pulse-ring border-white/30 bg-ink-800"
            : done
              ? "border-white/15 bg-ink-800"
              : status === "failed"
                ? "border-rose-500/50 bg-rose-500/10"
                : "border-white/[0.06] bg-ink-900/60"
        }`}
        style={{
          ["--ring-color" as string]: info.hex,
          boxShadow: done ? `0 0 24px -6px ${info.hex}88` : undefined,
        }}
      >
        <Icon className={`h-6 w-6 transition-colors ${status === "waiting" || status === "skipped" ? "text-slate-600" : info.text}`} />
        <span className="absolute -right-1.5 -top-1.5 grid h-5 w-5 place-items-center rounded-full border border-ink-950 bg-ink-800">
          {running && <Loader2 className="h-3 w-3 animate-spin text-cyan-300" />}
          {done && <Check className="h-3 w-3 text-emerald-300" />}
          {status === "failed" && <X className="h-3 w-3 text-rose-300" />}
          {status === "skipped" && <Minus className="h-3 w-3 text-slate-400" />}
        </span>
        {count && count > 1 ? (
          <span className="absolute -bottom-1.5 -right-1.5 rounded-full bg-ink-700 px-1.5 text-[10px] font-semibold text-slate-200 ring-1 ring-white/10">
            ×{count}
          </span>
        ) : null}
      </div>
      <div>
        <div className={`text-xs font-semibold ${status === "waiting" ? "text-slate-500" : "text-slate-100"}`}>{info.label}</div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500">{status === "waiting" ? "queued" : status}</div>
      </div>
    </div>
  );
}

function Connector({ active, done }: { active: boolean; done: boolean }) {
  return (
    <svg className="mt-7 hidden h-2 min-w-6 flex-1 md:block" preserveAspectRatio="none" viewBox="0 0 100 4">
      <line x1="0" y1="2" x2="100" y2="2" stroke={done ? "var(--color-ink-600)" : "var(--color-ink-700)"} strokeWidth="2" />
      {active && (
        <line x1="0" y1="2" x2="100" y2="2" stroke="var(--color-cyan-400)" strokeWidth="2" strokeDasharray="6 10" className="animate-flow" />
      )}
      {done && !active && <line x1="0" y1="2" x2="100" y2="2" stroke="color-mix(in oklab, var(--color-cyan-400) 35%, transparent)" strokeWidth="2" />}
    </svg>
  );
}

export function PipelineGraph({ run, maxRevisions }: { run: RunState; maxRevisions: number }) {
  const by = (agent: string) => run.stages.filter((s) => s.agent === agent);
  const verifier = by("Verifier");
  const refiner = by("Refiner");
  const evaluator: NodeStatus = run.evaluatorRunning
    ? "running"
    : run.result
      ? run.result.evaluator_error ? "failed" : "completed"
      : "waiting";
  const loopStatus = statusOf([...verifier, ...refiner].sort((a, b) => (a.started_at_utc ?? "").localeCompare(b.started_at_utc ?? "")));
  const nodes: { key: string; status: NodeStatus }[] = [
    { key: "Extractor", status: statusOf(by("Extractor")) },
    { key: "TermLookup", status: statusOf(by("TermLookup")) },
    { key: "Simplifier", status: statusOf(by("Simplifier")) },
    { key: "Loop", status: refiner.length || verifier.length ? (run.pipeline ? "completed" : loopStatus) : "waiting" },
    { key: "Readability", status: statusOf(by("Readability")) },
    { key: "Evaluator", status: evaluator },
  ];
  // Older recorded runs predate the Term Lookup stage.
  const visible = nodes.filter((n) => !(n.key === "TermLookup" && n.status === "waiting" && run.stages.some((s) => s.agent === "Simplifier")));
  const lastVerdict = verifier[verifier.length - 1]?.parsed_output?.overall_decision as string | undefined;

  return (
    <div className="glass grid-bg relative overflow-hidden p-5">
      <div className="flex flex-wrap items-start justify-center gap-y-6 md:flex-nowrap md:justify-between">
        {visible.map((node, index) => {
          const next = visible[index + 1];
          const connector = next ? (
            <Connector active={node.status === "completed" && next.status === "running"} done={node.status === "completed" || node.status === "skipped"} />
          ) : null;
          if (node.key !== "Loop")
            return (
              <div key={node.key} className="contents">
                <Node agent={node.key} status={node.status} />
                {connector}
              </div>
            );
          return (
            <div key="loop" className="contents">
              <div className="relative rounded-2xl border border-dashed border-white/10 px-3 pb-3 pt-6">
                <div className="absolute left-1/2 top-1 -translate-x-1/2 whitespace-nowrap text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  Verify ⇄ Refine loop
                </div>
                <div className="flex items-start gap-1">
                  <Node agent="Verifier" status={statusOf(verifier)} count={verifier.length} compact />
                  <div className="mt-4 flex flex-col items-center text-slate-500">
                    <RefreshCcw className={`h-4 w-4 ${refiner.some((r) => r.status === "running") ? "animate-spin text-pink-300" : ""}`} />
                    <span className="mt-1 text-[10px] font-medium">
                      {refiner.length}/{maxRevisions}
                    </span>
                  </div>
                  <Node agent="Refiner" status={refiner.length ? statusOf(refiner) : verifier.length && lastVerdict === "accept" ? "skipped" : "waiting"} count={refiner.length} compact />
                </div>
                {lastVerdict && (
                  <div
                    className={`mt-2 rounded-lg px-2 py-1 text-center text-[11px] font-semibold ${
                      lastVerdict === "accept" ? "bg-emerald-400/10 text-emerald-300" : "bg-amber-400/10 text-amber-300"
                    }`}
                  >
                    {lastVerdict === "accept" ? "✓ Accepted" : "↻ Revision requested"}
                  </div>
                )}
              </div>
              {connector}
            </div>
          );
        })}
      </div>
    </div>
  );
}
