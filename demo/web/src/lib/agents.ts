import { BookOpenText, PenLine, Scale, ScanText, ShieldCheck, Sparkles, Wand2, type LucideIcon } from "lucide-react";

export interface AgentInfo {
  key: string;
  label: string;
  role: string;
  icon: LucideIcon;
  /** Tailwind text color class and raw hex for glows. */
  text: string;
  hex: string;
}

export const AGENTS: Record<string, AgentInfo> = {
  Extractor: { key: "Extractor", label: "Extractor", role: "Pulls every checkable fact, with verbatim evidence", icon: ScanText, text: "text-cyan-300", hex: "#22d3ee" },
  TermLookup: { key: "TermLookup", label: "Term Lookup", role: "Grounds jargon in UMLS definitions", icon: BookOpenText, text: "text-sky-300", hex: "#38bdf8" },
  Simplifier: { key: "Simplifier", label: "Simplifier", role: "Drafts a plain-language explanation", icon: Wand2, text: "text-violet-300", hex: "#a78bfa" },
  Verifier: { key: "Verifier", label: "Verifier", role: "Audits every sentence against the source", icon: ShieldCheck, text: "text-amber-300", hex: "#fbbf24" },
  Refiner: { key: "Refiner", label: "Refiner", role: "Makes minimal, targeted fixes", icon: PenLine, text: "text-pink-300", hex: "#f472b6" },
  Readability: { key: "Readability", label: "Readability", role: "Polishes to an 8th-grade reading level", icon: Sparkles, text: "text-emerald-300", hex: "#34d399" },
  Evaluator: { key: "Evaluator", label: "Fact-Check", role: "Independent claim-by-claim faithfulness audit", icon: Scale, text: "text-teal-200", hex: "#5eead4" },
};

export function agentInfo(agent: string): AgentInfo {
  return AGENTS[agent] ?? AGENTS.Extractor;
}

export const CLASS_STYLE: Record<string, { label: string; chip: string; underline: string; bg: string }> = {
  supported: { label: "Supported", chip: "bg-emerald-400/10 text-emerald-300 ring-emerald-400/30", underline: "decoration-emerald-400/60", bg: "bg-emerald-400/10" },
  uncertain: { label: "Uncertain", chip: "bg-slate-400/10 text-slate-300 ring-slate-400/30", underline: "decoration-slate-400/70", bg: "bg-slate-400/10" },
  unsupported: { label: "Unsupported", chip: "bg-amber-400/10 text-amber-300 ring-amber-400/30", underline: "decoration-amber-400", bg: "bg-amber-400/15" },
  contradicted: { label: "Contradicted", chip: "bg-rose-500/10 text-rose-300 ring-rose-500/30", underline: "decoration-rose-500", bg: "bg-rose-500/15" },
};

export const SEVERITY: Record<string, number> = { supported: 0, uncertain: 1, unsupported: 2, contradicted: 3 };
