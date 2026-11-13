import { AlertTriangle, ArrowRight, CircleSlash, FileWarning, Quote } from "lucide-react";
import { useMemo, useState } from "react";
import { friendlyError, splitSimplifier, tryParseJson, wordDiff } from "../lib/text";
import type { Stage } from "../lib/types";
import { Markdown } from "./Markdown";
import { Chip } from "./ui";

const FACT_TYPE: Record<string, string> = {
  diagnosis: "bg-rose-400/10 text-rose-300 ring-rose-400/30",
  medication: "bg-violet-400/10 text-violet-300 ring-violet-400/30",
  lab_value: "bg-cyan-400/10 text-cyan-300 ring-cyan-400/30",
  procedure: "bg-sky-400/10 text-sky-300 ring-sky-400/30",
  instruction: "bg-emerald-400/10 text-emerald-300 ring-emerald-400/30",
  demographics: "bg-slate-400/10 text-slate-300 ring-slate-400/30",
};

const RELEVANCE: Record<string, string> = {
  required: "bg-emerald-400/10 text-emerald-300 ring-emerald-400/30",
  borderline: "bg-sky-400/10 text-sky-300 ring-sky-400/30",
  unnecessary: "bg-slate-400/10 text-slate-400 ring-slate-400/30",
  inappropriate: "bg-rose-500/10 text-rose-300 ring-rose-500/30",
};

interface Fact {
  fact_id: string;
  fact_text: string;
  fact_type: string;
  source_evidence: string[];
}

export function summarize(stage: Stage): string {
  if (stage.status === "failed") return "Failed";
  switch (stage.agent) {
    case "Extractor": {
      const facts = tryParseJson<Fact[]>(stage.output);
      return facts ? `${facts.length} facts extracted` : "Extracting facts";
    }
    case "TermLookup": {
      const defs = stage.parsed_output;
      if (stage.status === "skipped" || defs?.status === "not_configured") return "Skipped · no UMLS key configured";
      return defs ? `${defs.entries?.length ?? 0} definitions · ${defs.unresolved?.length ?? 0} unresolved` : "Looking up terms";
    }
    case "Verifier": {
      const v = stage.parsed_output;
      if (!v) return "Auditing draft";
      const issues = (v.unsupported_claims?.length ?? 0) + (v.omissions?.length ?? 0);
      return v.overall_decision === "accept" ? "Accepted the draft" : `Revision requested · ${issues} issue${issues === 1 ? "" : "s"}`;
    }
    case "Refiner":
      return "Applied targeted fixes";
    case "Simplifier":
      return "Plain-language draft written";
    case "Readability":
      return "Final explanation polished";
    default:
      return stage.status;
  }
}

function ExtractorView({ output }: { output: string }) {
  const facts = tryParseJson<Fact[]>(output) ?? [];
  const counts = facts.reduce<Record<string, number>>((acc, f) => ({ ...acc, [f.fact_type]: (acc[f.fact_type] ?? 0) + 1 }), {});
  if (!facts.length)
    return (
      <p className="flex items-center gap-2 text-sm text-slate-400">
        <CircleSlash className="h-4 w-4 shrink-0" />
        The Extractor returned no patient-level facts for this source, so later agents worked from the source text alone.
      </p>
    );
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {Object.entries(counts).map(([type, n]) => (
          <Chip key={type} className={FACT_TYPE[type]}>
            {type.replace("_", " ")} · {n}
          </Chip>
        ))}
      </div>
      <div className="scroll-thin grid max-h-96 gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
        {facts.map((fact) => (
          <div key={fact.fact_id} className="rounded-xl border border-white/[0.06] bg-ink-850 p-3" title={fact.source_evidence.join("\n")}>
            <div className="mb-1 flex items-center gap-2">
              <span className="font-mono text-[10px] text-slate-500">{fact.fact_id}</span>
              <Chip className={FACT_TYPE[fact.fact_type]}>{fact.fact_type.replace("_", " ")}</Chip>
            </div>
            <p className="text-sm leading-snug text-slate-200">{fact.fact_text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function TermLookupView({ stage }: { stage: Stage }) {
  const defs = stage.parsed_output ?? {};
  const entries: any[] = defs.entries ?? [];
  if (!entries.length)
    return (
      <p className="flex items-center gap-2 text-sm text-slate-400">
        <CircleSlash className="h-4 w-4" />
        {defs.status === "not_configured"
          ? "No UMLS key on this server, so the stage was skipped with no model call."
          : "No reference definitions were found."}
      </p>
    );
  return (
    <div className="space-y-2">
      {entries.map((e) => (
        <div key={`${e.term}-${e.source_id}`} className="rounded-xl border border-white/[0.06] bg-ink-850 p-3">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <span className="font-semibold text-slate-100">{e.term}</span>
            <Chip className="bg-sky-400/10 text-sky-300 ring-sky-400/30">
              {e.source} {e.source_id}
            </Chip>
          </div>
          <p className="text-sm text-slate-300">{e.definition}</p>
        </div>
      ))}
    </div>
  );
}

function SimplifierView({ output }: { output: string }) {
  const { body, flags } = splitSimplifier(output);
  return (
    <div className="space-y-3">
      <Markdown text={body} className="text-sm text-slate-300" />
      {flags && (
        <div className="rounded-xl border border-violet-400/20 bg-violet-400/5 p-3 text-sm">
          <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-violet-300">
            <FileWarning className="h-3.5 w-3.5" /> Review flags raised by the Simplifier
          </div>
          <Markdown text={flags} className="text-slate-300" />
        </div>
      )}
    </div>
  );
}

function VerifierView({ verdict }: { verdict: any }) {
  if (!verdict) return null;
  const accept = verdict.overall_decision === "accept";
  const unsupported: any[] = verdict.unsupported_claims ?? [];
  const omissions: any[] = verdict.omissions ?? [];
  const sentences: any[] = verdict.sentences ?? [];
  return (
    <div className="space-y-4">
      <div className={`flex items-start gap-3 rounded-xl p-3 ${accept ? "bg-emerald-400/10" : "bg-amber-400/10"}`}>
        <div className={`rounded-lg px-2.5 py-1 text-sm font-bold tracking-wide ${accept ? "bg-emerald-400 text-on-accent" : "bg-amber-400 text-on-accent"}`}>
          {accept ? "ACCEPT" : "REVISE"}
        </div>
        <p className="text-sm text-slate-200">{verdict.explanation_accuracy?.reason}</p>
      </div>

      {[...unsupported.map((c) => ({ kind: c.issue_type, quote: c.draft_quote, source: c.source_quote, fix: c.instruction_to_refiner })),
        ...omissions.map((o) => ({ kind: "omission", quote: o.missing_fact, source: (o.source_evidence ?? []).join(" "), fix: o.instruction_to_refiner }))].map(
        (issue, i) => (
          <div key={i} className="rounded-xl border border-amber-400/20 bg-ink-850 p-3">
            <div className="mb-2 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-300" />
              <Chip className={issue.kind === "contradicted" ? "bg-rose-500/10 text-rose-300 ring-rose-500/30" : "bg-amber-400/10 text-amber-300 ring-amber-400/30"}>
                {issue.kind}
              </Chip>
            </div>
            <p className="text-sm text-slate-100">“{issue.quote}”</p>
            {issue.source && (
              <p className="mt-2 flex gap-2 border-l-2 border-white/10 pl-3 text-xs text-slate-400">
                <Quote className="h-3 w-3 shrink-0" />
                {issue.source}
              </p>
            )}
            {issue.fix && (
              <p className="mt-2 flex items-start gap-2 text-xs text-pink-200">
                <ArrowRight className="mt-0.5 h-3 w-3 shrink-0" /> To Refiner: {issue.fix}
              </p>
            )}
          </div>
        ),
      )}

      {sentences.length > 0 && (
        <details className="group rounded-xl border border-white/[0.06] bg-ink-850">
          <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Sentence-by-sentence relevance ({sentences.length})
          </summary>
          <div className="scroll-thin max-h-72 space-y-1.5 overflow-y-auto px-3 pb-3">
            {sentences.map((s) => (
              <div key={s.sentence_number} className="flex gap-2 text-sm">
                <Chip className={`${RELEVANCE[s.relevance] ?? ""} h-fit shrink-0`}>{s.relevance}</Chip>
                <span className="text-slate-300" title={s.reason}>
                  {s.sentence_text}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

const CONTEXT_WORDS = 14;

/** Shorten an unchanged run to its edges so the Refiner's edits stand out. */
function collapse(text: string, keepStart: boolean, keepEnd: boolean): { head: string; hidden: number; tail: string } | null {
  const words = text.split(/(\s+)/);
  const count = words.filter((w) => w.trim()).length;
  if (count <= CONTEXT_WORDS * 2 + 10) return null;
  const take = (n: number, fromEnd: boolean) => {
    const out: string[] = [];
    let seen = 0;
    const seq = fromEnd ? [...words].reverse() : words;
    for (const w of seq) {
      if (seen >= n) break;
      out.push(w);
      if (w.trim()) seen++;
    }
    return (fromEnd ? out.reverse() : out).join("");
  };
  const head = keepStart ? take(CONTEXT_WORDS, false) : "";
  const tail = keepEnd ? take(CONTEXT_WORDS, true) : "";
  return { head, tail, hidden: count - (keepStart ? CONTEXT_WORDS : 0) - (keepEnd ? CONTEXT_WORDS : 0) };
}

function RefinerView({ before, after }: { before: string; after: string }) {
  const parts = useMemo(() => wordDiff(before, after), [before, after]);
  const [full, setFull] = useState(false);
  const added = parts.filter((p) => p.kind === "add").reduce((n, p) => n + p.text.trim().split(/\s+/).filter(Boolean).length, 0);
  const removed = parts.filter((p) => p.kind === "del").reduce((n, p) => n + p.text.trim().split(/\s+/).filter(Boolean).length, 0);
  return (
    <div>
      <div className="mb-2 flex gap-2 text-xs">
        <Chip className="bg-emerald-400/10 text-emerald-300 ring-emerald-400/30">+{added} words</Chip>
        <Chip className="bg-rose-500/10 text-rose-300 ring-rose-500/30">−{removed} words</Chip>
        <span className="text-slate-500">Everything else is left untouched.</span>
        <button onClick={() => setFull(!full)} className="ml-auto text-slate-500 hover:text-slate-300">
          {full ? "Show changes only" : "Show full text"}
        </button>
      </div>
      <div className="scroll-thin max-h-96 overflow-y-auto whitespace-pre-wrap rounded-xl bg-ink-850 p-3 text-sm leading-relaxed text-slate-400">
        {parts.map((p, i) => {
          if (p.kind === "same") {
            const short = full ? null : collapse(p.text, i > 0, i < parts.length - 1);
            if (!short) return <span key={i}>{p.text}</span>;
            return (
              <span key={i}>
                {short.head}
                <span className="mx-1 inline-block rounded-md bg-white/5 px-2 py-0.5 text-[11px] text-slate-500">⋯ {short.hidden} words unchanged ⋯</span>
                {short.tail}
              </span>
            );
          }
          return p.kind === "add" ? (
            <span key={i} className="rounded bg-emerald-400/15 text-emerald-200">
              {p.text}
            </span>
          ) : (
            <span key={i} className="rounded bg-rose-500/15 text-rose-300 line-through decoration-rose-400/70">
              {p.text}
            </span>
          );
        })}
      </div>
    </div>
  );
}

export function StageBody({ stage, previousDraft }: { stage: Stage; previousDraft: string | null }) {
  if (stage.error && stage.status === "failed") return <p className="text-sm text-rose-300">{friendlyError(stage.error)}</p>;
  switch (stage.agent) {
    case "Extractor":
      return <ExtractorView output={stage.output} />;
    case "TermLookup":
      return <TermLookupView stage={stage} />;
    case "Simplifier":
      return <SimplifierView output={stage.output} />;
    case "Verifier":
      return <VerifierView verdict={stage.parsed_output ?? tryParseJson(stage.output)} />;
    case "Refiner":
      return previousDraft ? <RefinerView before={previousDraft} after={stage.output} /> : <Markdown text={stage.output} className="text-sm" />;
    default:
      return <Markdown text={stage.output} className="text-sm text-slate-200" />;
  }
}
