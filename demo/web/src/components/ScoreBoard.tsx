import { BookOpen, ShieldAlert, ShieldCheck, Wrench } from "lucide-react";
import type { RunState } from "../lib/runState";
import type { ResultEvent } from "../lib/types";
import { fmt } from "./ui";

function Ring({ value, target, color }: { value: number | null; target: number; color: string }) {
  const r = 42;
  const c = 2 * Math.PI * r;
  const pct = value === null ? 0 : Math.max(0, Math.min(100, value)) / 100;
  const t = target / 100;
  return (
    <svg viewBox="0 0 100 100" className="h-28 w-28 -rotate-90">
      <circle cx="50" cy="50" r={r} fill="none" stroke="var(--color-ink-700)" strokeWidth="9" />
      <circle
        cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round"
        strokeDasharray={`${c * pct} ${c}`} style={{ transition: "stroke-dasharray 1.2s ease-out", filter: `drop-shadow(0 0 6px color-mix(in oklab, ${color} 55%, transparent))` }}
      />
      <line
        x1={50 + (r - 7) * Math.cos(2 * Math.PI * t)} y1={50 + (r - 7) * Math.sin(2 * Math.PI * t)}
        x2={50 + (r + 7) * Math.cos(2 * Math.PI * t)} y2={50 + (r + 7) * Math.sin(2 * Math.PI * t)}
        stroke="var(--color-slate-200)" strokeWidth="1.5"
      />
    </svg>
  );
}

function Tile({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`glass flex flex-col p-5 ${className}`}>{children}</div>;
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">{children}</div>;
}

/** Issues the Verifier caught during the loop, and how many passes fixed them. */
export function loopStory(run: RunState) {
  const verdicts = run.stages.filter((s) => s.agent === "Verifier" && s.parsed_output).map((s) => s.parsed_output);
  const caught = verdicts
    .slice(0, -1)
    .reduce((n, v) => n + (v.unsupported_claims?.length ?? 0) + (v.omissions?.length ?? 0), 0);
  return { caught, revisions: run.stages.filter((s) => s.agent === "Refiner").length, checks: verdicts.length };
}

export function ScoreBoard({ result, run }: { result: ResultEvent; run: RunState }) {
  const fidelity = result.factual?.factual_fidelity_estimate ?? null;
  const grade = result.readability?.flesch_kincaid_grade ?? null;
  const flagged = result.claims.filter((c) => c.clinically_meaningful && (c.classification === "unsupported" || c.classification === "contradicted"));
  const counts = ["supported", "uncertain", "unsupported", "contradicted"].map((k) => ({
    k,
    n: result.claims.filter((c) => c.classification === k).length,
  }));
  const total = Math.max(1, result.claims.length);
  const story = loopStory(run);
  const barColor: Record<string, string> = { supported: "bg-emerald-400", uncertain: "bg-slate-400", unsupported: "bg-amber-400", contradicted: "bg-rose-500" };

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Tile>
        <Label>Faithfulness</Label>
        <div className="mt-2 flex items-center gap-4">
          <div className="relative">
            <Ring value={fidelity} target={85} color="var(--color-emerald-400)" />
            <div className="absolute inset-0 grid place-items-center text-2xl font-bold text-slate-50">{fmt(fidelity, 0, "%")}</div>
          </div>
          <div className="text-sm text-slate-400">
            <div className="text-slate-200">
              {result.factual?.supported_claims ?? 0}/{result.factual?.claims_assessed ?? 0} claims
            </div>
            supported by the source
            <div className="mt-1 text-xs text-slate-500">target ≥ 85%</div>
          </div>
        </div>
        <div className="mt-4 flex h-2 overflow-hidden rounded-full bg-ink-700">
          {counts.map(({ k, n }) => (
            <div key={k} className={`${barColor[k]} transition-all duration-1000`} style={{ width: `${(n / total) * 100}%` }} title={`${k}: ${n}`} />
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-3 text-[11px] text-slate-500">
          {counts.map(({ k, n }) => (
            <span key={k}>
              <span className={`mr-1 inline-block h-2 w-2 rounded-full ${barColor[k]}`} />
              {k} {n}
            </span>
          ))}
        </div>
      </Tile>

      <Tile>
        <Label>Reading level</Label>
        <div className="mt-3 flex items-baseline gap-3">
          <span className="text-2xl font-semibold text-slate-500 line-through decoration-slate-600">{fmt(result.source_grade)}</span>
          <span className="text-slate-500">→</span>
          <span className={`text-4xl font-bold ${result.readability?.meets_target ? "text-emerald-300" : "text-amber-300"}`}>{fmt(grade)}</span>
        </div>
        <div className="mt-1 text-sm text-slate-400">Flesch–Kincaid grade · source → final</div>
        <div className="relative mt-5 h-2 rounded-full bg-gradient-to-r from-emerald-400/70 via-amber-400/70 to-rose-500/70">
          {[result.source_grade, grade].map((g, i) =>
            g === null ? null : (
              <div
                key={i}
                className={`absolute -top-1.5 h-5 w-1.5 rounded-full ${i ? "bg-white shadow-[0_0_10px_var(--color-white)]" : "bg-slate-500"}`}
                style={{ left: `${Math.min(100, (g / 18) * 100)}%`, transition: "left 1s" }}
              />
            ),
          )}
          <div className="absolute -bottom-5 text-[10px] text-slate-500" style={{ left: `${(8 / 18) * 100}%` }}>
            ▲ grade 8 target
          </div>
        </div>
        <div className="mt-7 text-xs text-slate-500">
          SMOG {fmt(result.readability?.smog_index)} · {result.readability?.word_count ?? "—"} words
        </div>
      </Tile>

      <Tile>
        <Label>Hallucination check</Label>
        <div className="mt-3 flex items-center gap-3">
          {flagged.length ? <ShieldAlert className="h-10 w-10 text-amber-300" /> : <ShieldCheck className="h-10 w-10 text-emerald-300" />}
          <div>
            <div className="text-4xl font-bold text-slate-50">{flagged.length}</div>
            <div className="text-sm text-slate-400">clinically meaningful flag{flagged.length === 1 ? "" : "s"}</div>
          </div>
        </div>
        <p className="mt-3 text-sm text-slate-400">
          {flagged.length
            ? "Flagged by the automated fact-check for human review. Not a confirmed error."
            : "The independent fact-check found no clinically meaningful unsupported claims."}
        </p>
      </Tile>

      <Tile>
        <Label>Self-correction</Label>
        <div className="mt-3 flex items-center gap-3">
          <Wrench className="h-10 w-10 text-pink-300" />
          <div>
            <div className="text-4xl font-bold text-slate-50">{story.revisions}</div>
            <div className="text-sm text-slate-400">refinement pass{story.revisions === 1 ? "" : "es"}</div>
          </div>
        </div>
        <p className="mt-3 text-sm text-slate-400">
          {story.revisions
            ? `The Verifier caught ${story.caught} issue${story.caught === 1 ? "" : "s"} before the patient saw them; the Refiner fixed them and the draft passed on check ${story.checks}.`
            : "The Verifier accepted the first draft, so no corrections were needed."}
        </p>
        <div className="mt-auto flex items-center gap-1.5 pt-3 text-xs text-slate-500">
          <BookOpen className="h-3.5 w-3.5" /> {run.stages.length + 1} agent calls in total
        </div>
      </Tile>
    </div>
  );
}
