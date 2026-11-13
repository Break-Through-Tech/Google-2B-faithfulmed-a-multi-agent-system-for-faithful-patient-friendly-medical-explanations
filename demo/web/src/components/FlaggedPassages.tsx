import { CheckCircle2, Flag } from "lucide-react";
import { CLASS_STYLE, SEVERITY } from "../lib/agents";
import { plain } from "../lib/text";
import type { ResultEvent } from "../lib/types";
import type { Selection } from "./EvidenceView";
import { Chip } from "./ui";

export function FlaggedPassages({ result, onSelect }: { result: ResultEvent; onSelect: (s: Selection) => void }) {
  const spanText = new Map(result.explanation_spans.map((s) => [s.span_id, s.text]));
  const sourceText = new Map(result.source_spans.map((s) => [s.span_id, s.text]));
  const flagged = result.claims
    .filter((c) => c.classification !== "supported")
    .sort((a, b) => SEVERITY[b.classification] - SEVERITY[a.classification] || Number(b.clinically_meaningful) - Number(a.clinically_meaningful));

  if (!flagged.length)
    return (
      <div className="glass flex items-center gap-3 p-5 text-sm text-slate-300">
        <CheckCircle2 className="h-5 w-5 text-emerald-300" /> Every claim in the final explanation traced back to the source.
      </div>
    );

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {flagged.map((claim) => {
        const style = CLASS_STYLE[claim.classification];
        return (
          <button
            key={claim.claim_id}
            onClick={() => claim.claim_span_id && onSelect({ kind: "explanation", id: claim.claim_span_id })}
            className="glass group p-4 text-left transition hover:border-white/20"
          >
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Flag className={`h-4 w-4 ${claim.classification === "contradicted" ? "text-rose-400" : "text-amber-300"}`} />
              <Chip className={style.chip}>{style.label}</Chip>
              {claim.clinically_meaningful && <Chip className="bg-white/5 text-slate-300 ring-white/10">clinically meaningful</Chip>}
              <span className="ml-auto text-[11px] text-slate-500 group-hover:text-cyan-300">show in context →</span>
            </div>
            <p className={`rounded-lg px-2 py-1.5 text-sm text-slate-100 ${style.bg}`}>
              “{plain(spanText.get(claim.claim_span_id ?? "") ?? claim.claim_text)}”
            </p>
            <p className="mt-2 text-xs text-slate-400">{claim.reason}</p>
            {claim.source_span_ids.length > 0 && (
              <p className="mt-2 border-l-2 border-white/10 pl-2 font-serif text-xs text-slate-500">
                Source: {claim.source_span_ids.map((id) => sourceText.get(id)).join(" ")}
              </p>
            )}
          </button>
        );
      })}
    </div>
  );
}
