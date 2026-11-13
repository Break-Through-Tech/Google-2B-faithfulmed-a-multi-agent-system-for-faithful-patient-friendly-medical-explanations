import { ChevronDown, Code2, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AGENTS, agentInfo } from "../lib/agents";
import type { RunState } from "../lib/runState";
import { seconds } from "../lib/text";
import type { Stage } from "../lib/types";
import { Markdown } from "./Markdown";
import { StageBody, summarize } from "./StageViews";

const TEXT_STAGES = new Set(["Simplifier", "Refiner", "Readability"]);

/** Live view of a stage's streamed tokens, pinned to the bottom. */
function StreamView({ text, prose }: { text: string; prose: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text]);
  if (!text)
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin" /> Waiting for the first tokens from Gemini…
      </div>
    );
  return (
    <div ref={ref} className="scroll-thin max-h-80 overflow-y-auto rounded-xl bg-ink-950/60 p-3">
      {prose ? (
        <div className="caret">
          <Markdown text={text} className="text-sm text-slate-200" />
        </div>
      ) : (
        <pre className="caret whitespace-pre-wrap break-words font-mono text-[11.5px] leading-relaxed text-cyan-100/80">
          {text.length > 6000 ? "…" + text.slice(-6000) : text}
        </pre>
      )}
    </div>
  );
}

function StageCard({
  stage,
  previousDraft,
  active,
  follow,
}: {
  stage: Stage;
  previousDraft: string | null;
  active: boolean;
  follow: boolean;
}) {
  const info = agentInfo(stage.agent);
  const Icon = info.icon;
  const running = stage.status === "running";
  const [open, setOpen] = useState(false);
  const [raw, setRaw] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const expanded = running || open;
  const duration = seconds(stage.started_at_utc, stage.completed_at_utc);

  useEffect(() => {
    if (active && follow) ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active, follow]);

  return (
    <div
      ref={ref}
      className={`glass animate-fade-up overflow-hidden transition-colors ${running ? "border-white/20" : ""}`}
      style={running ? { boxShadow: `0 0 0 1px ${info.hex}40, 0 0 40px -12px ${info.hex}80` } : undefined}
    >
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-3 px-4 py-3 text-left" disabled={running}>
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-ink-800 ring-1 ring-white/10">
          <Icon className={`h-[18px] w-[18px] ${info.text}`} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-100">{info.label}</span>
            {stage.iteration ? <span className="text-xs text-slate-500">pass {stage.iteration}</span> : null}
          </div>
          <div className="truncate text-xs text-slate-400">{running ? info.role : summarize(stage)}</div>
        </div>
        {running ? (
          <span className="flex items-center gap-1.5 text-xs font-medium text-cyan-300">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300" /> streaming
          </span>
        ) : (
          <span className="text-xs tabular-nums text-slate-500">{duration !== null ? `${duration.toFixed(1)}s` : ""}</span>
        )}
        {!running && <ChevronDown className={`h-4 w-4 text-slate-500 transition-transform ${open ? "rotate-180" : ""}`} />}
      </button>
      {expanded && (
        <div className="border-t border-white/[0.06] px-4 py-4">
          {running ? (
            <StreamView text={stage.output} prose={TEXT_STAGES.has(stage.agent)} />
          ) : (
            <>
              {raw ? (
                <pre className="scroll-thin max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-xl bg-ink-950/60 p-3 font-mono text-[11.5px] text-slate-300">
                  {stage.output || "(no output)"}
                </pre>
              ) : (
                <StageBody stage={stage} previousDraft={previousDraft} />
              )}
              <button onClick={() => setRaw(!raw)} className="mt-3 flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300">
                <Code2 className="h-3.5 w-3.5" /> {raw ? "Show formatted view" : "Show raw model output"}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function Timeline({ run, follow }: { run: RunState; follow: boolean }) {
  // The draft each Refiner pass received, for its diff view.
  const draftBefore = (index: number): string | null => {
    for (let i = index - 1; i >= 0; i--) {
      const s = run.stages[i];
      if (s.agent === "Simplifier" || s.agent === "Refiner") return s.output;
    }
    return null;
  };
  const Eval = AGENTS.Evaluator;
  return (
    <div className="space-y-3">
      {run.stages.map((stage, index) => (
        <StageCard
          key={stage.stage_id}
          stage={stage}
          previousDraft={stage.agent === "Refiner" ? draftBefore(index) : null}
          active={run.activeStageId === stage.stage_id}
          follow={follow}
        />
      ))}
      {(run.evaluatorRunning || run.evaluatorText) && (
        <div
          className="glass animate-fade-up overflow-hidden"
          style={run.evaluatorRunning ? { boxShadow: `0 0 0 1px ${Eval.hex}40, 0 0 40px -12px ${Eval.hex}80` } : undefined}
        >
          <div className="flex items-center gap-3 px-4 py-3">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-ink-800 ring-1 ring-white/10">
              <Eval.icon className={`h-[18px] w-[18px] ${Eval.text}`} />
            </div>
            <div className="flex-1">
              <div className="font-semibold text-slate-100">Independent Fact-Check</div>
              <div className="text-xs text-slate-400">
                {run.evaluatorRunning ? Eval.role : `${run.result?.claims.length ?? 0} atomic claims checked against the source`}
              </div>
            </div>
          </div>
          {run.evaluatorRunning && (
            <div className="border-t border-white/[0.06] px-4 py-4">
              <StreamView text={run.evaluatorText} prose={false} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
