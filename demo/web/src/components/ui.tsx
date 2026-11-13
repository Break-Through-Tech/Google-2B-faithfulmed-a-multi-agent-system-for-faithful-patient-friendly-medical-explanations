import type { ReactNode } from "react";

export function Chip({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${className}`}>
      {children}
    </span>
  );
}

export function SectionTitle({ kicker, title, right }: { kicker: string; title: string; right?: ReactNode }) {
  return (
    <div className="mb-3 flex items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-300/80">{kicker}</div>
        <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
      </div>
      {right}
    </div>
  );
}

export function fmt(value: number | null | undefined, digits = 1, suffix = ""): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : `${value.toFixed(digits)}${suffix}`;
}
