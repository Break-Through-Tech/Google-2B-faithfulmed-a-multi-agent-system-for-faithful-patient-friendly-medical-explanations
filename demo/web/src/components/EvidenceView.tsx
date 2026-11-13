import { Link2, MousePointerClick, Quote } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { CLASS_STYLE, SEVERITY } from "../lib/agents";
import { plain } from "../lib/text";
import type { Claim, ResultEvent, Span } from "../lib/types";
import { Chip } from "./ui";

export type Selection = { kind: "explanation" | "source"; id: string } | null;

function worst(claims: Claim[]): string | null {
  if (!claims.length) return null;
  return claims.reduce((a, c) => (SEVERITY[c.classification] > SEVERITY[a.classification] ? c : a)).classification;
}

function isHeading(text: string) {
  return /^#{1,6}\s/.test(text);
}
function isBullet(text: string) {
  return /^[-*+]\s/.test(text);
}

export function EvidenceView({
  result,
  selection,
  onSelect,
}: {
  result: ResultEvent;
  selection: Selection;
  onSelect: (s: Selection) => void;
}) {
  const claimsBySpan = useMemo(() => {
    const map = new Map<string, Claim[]>();
    for (const claim of result.claims) {
      if (!claim.claim_span_id) continue;
      map.set(claim.claim_span_id, [...(map.get(claim.claim_span_id) ?? []), claim]);
    }
    return map;
  }, [result.claims]);

  const selectedClaims = useMemo(() => {
    if (!selection) return [];
    return selection.kind === "explanation"
      ? claimsBySpan.get(selection.id) ?? []
      : result.claims.filter((c) => c.source_span_ids.includes(selection.id));
  }, [selection, claimsBySpan, result.claims]);

  const litSource = new Set(selectedClaims.flatMap((c) => c.source_span_ids));
  const litExplanation = new Set(selectedClaims.map((c) => c.claim_span_id));
  const sourceRefs = useRef<Record<string, HTMLSpanElement | null>>({});
  const explRefs = useRef<Record<string, HTMLSpanElement | null>>({});

  useEffect(() => {
    if (!selection) return;
    const targets = selection.kind === "explanation" ? [...litSource] : [...litExplanation];
    const refs = selection.kind === "explanation" ? sourceRefs.current : explRefs.current;
    const first = targets.map((id) => (id ? refs[id] : null)).find(Boolean);
    first?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selection]);

  const renderExplanation = (spans: Span[]) => {
    const blocks: React.ReactNode[] = [];
    let paragraph: React.ReactNode[] = [];
    let list: React.ReactNode[] = [];
    const flush = () => {
      if (paragraph.length) blocks.push(<p key={`p${blocks.length}`} className="my-2 leading-relaxed">{paragraph}</p>);
      if (list.length) blocks.push(<ul key={`u${blocks.length}`} className="my-2 list-disc space-y-1 pl-5 marker:text-cyan-400/60">{list}</ul>);
      paragraph = [];
      list = [];
    };
    for (const span of spans) {
      if (/^(-{3,}|\*{3,})$/.test(span.text)) {
        flush();
        continue;
      }
      const claims = claimsBySpan.get(span.span_id) ?? [];
      const level = worst(claims);
      const style = level ? CLASS_STYLE[level] : null;
      const selected = selection?.kind === "explanation" && selection.id === span.span_id;
      const lit = litExplanation.has(span.span_id);
      const node = (
        <span
          key={span.span_id}
          ref={(el) => {
            explRefs.current[span.span_id] = el;
          }}
          onClick={() => claims.length && onSelect(selected ? null : { kind: "explanation", id: span.span_id })}
          className={`rounded px-0.5 transition-colors ${
            claims.length ? `cursor-pointer underline decoration-2 underline-offset-4 ${style?.underline}` : ""
          } ${selected || lit ? "bg-cyan-400/20 text-white" : level && level !== "supported" ? style?.bg : "hover:bg-white/5"}`}
        >
          {plain(span.text)}{" "}
        </span>
      );
      if (isHeading(span.text)) {
        flush();
        blocks.push(<h4 key={span.span_id} className="mb-1 mt-4 font-semibold text-slate-50 first:mt-0">{node}</h4>);
      } else if (isBullet(span.text)) {
        if (paragraph.length) flush();
        list.push(<li key={span.span_id}>{node}</li>);
      } else {
        if (list.length) flush();
        paragraph.push(node);
      }
    }
    flush();
    return blocks;
  };

  return (
    <div className="glass overflow-hidden">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-white/[0.06] px-5 py-3 text-xs text-slate-400">
        <span className="flex items-center gap-1.5 text-slate-300">
          <MousePointerClick className="h-3.5 w-3.5 text-cyan-300" /> Click any underlined sentence to see the evidence behind it
        </span>
        {Object.entries(CLASS_STYLE).map(([key, s]) => (
          <span key={key} className={`underline decoration-2 underline-offset-4 ${s.underline}`}>
            {s.label}
          </span>
        ))}
      </div>
      <div className="grid lg:grid-cols-2">
        <div className="border-b border-white/[0.06] lg:border-b-0 lg:border-r">
          <div className="px-5 pt-4 text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-300/80">Patient explanation</div>
          <div className="scroll-thin max-h-[34rem] overflow-y-auto px-5 pb-5 text-[15px] text-slate-200">
            {renderExplanation(result.explanation_spans)}
          </div>
        </div>
        <div>
          <div className="px-5 pt-4 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">Original clinical source</div>
          <div className="scroll-thin max-h-[34rem] overflow-y-auto px-5 pb-5 pt-2 font-serif text-[13.5px] leading-relaxed text-slate-400">
            {result.source_spans.map((span) => {
              const lit = litSource.has(span.span_id);
              const selected = selection?.kind === "source" && selection.id === span.span_id;
              return (
                <span
                  key={span.span_id}
                  ref={(el) => {
                    sourceRefs.current[span.span_id] = el;
                  }}
                  onClick={() => onSelect(selected ? null : { kind: "source", id: span.span_id })}
                  className={`cursor-pointer rounded px-0.5 transition-colors duration-300 ${
                    lit || selected ? "bg-cyan-400/20 text-slate-50 ring-1 ring-cyan-400/40" : "hover:bg-white/5"
                  }`}
                >
                  {span.text}{" "}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      {selectedClaims.length > 0 && (
        <div className="animate-fade-up border-t border-white/[0.06] bg-ink-850/80 px-5 py-4">
          <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-300/80">
            <Link2 className="h-3.5 w-3.5" /> Evidence attribution · {selectedClaims.length} claim{selectedClaims.length === 1 ? "" : "s"}
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {selectedClaims.map((claim) => (
              <div key={claim.claim_id} className="rounded-xl border border-white/[0.06] bg-ink-900 p-3">
                <div className="mb-1.5 flex flex-wrap items-center gap-2">
                  <Chip className={CLASS_STYLE[claim.classification]?.chip}>{CLASS_STYLE[claim.classification]?.label}</Chip>
                  {claim.clinically_meaningful && <Chip className="bg-white/5 text-slate-300 ring-white/10">clinically meaningful</Chip>}
                  <span className="font-mono text-[10px] text-slate-500">
                    {claim.claim_id} · {claim.claim_span_id} ← {claim.source_span_ids.join(", ") || "no source span"}
                  </span>
                </div>
                <p className="text-sm text-slate-100">{claim.claim_text}</p>
                <p className="mt-1.5 flex gap-1.5 text-xs text-slate-400">
                  <Quote className="mt-0.5 h-3 w-3 shrink-0" />
                  {claim.reason}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
