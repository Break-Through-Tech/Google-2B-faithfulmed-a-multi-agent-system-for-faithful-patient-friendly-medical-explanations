/** Parsing helpers for agent outputs. */

export function tryParseJson<T = any>(text: string): T | null {
  const stripped = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try {
    return JSON.parse(stripped) as T;
  } catch {
    return null;
  }
}

/** The complete objects so far in a streaming JSON array of objects. */
export function completedObjects<T = any>(text: string): T[] {
  const items: T[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === "{") {
      if (depth++ === 0) start = i;
    } else if (ch === "}" && depth > 0 && --depth === 0) {
      try {
        items.push(JSON.parse(text.slice(start, i + 1)) as T);
      } catch {
        // Not a complete object after all; the final parse covers it.
      }
    }
  }
  return items;
}

/** Split the Simplifier's two-section output. */
export function splitSimplifier(text: string): { body: string; flags: string } {
  const flagsAt = text.search(/REVIEW FLAGS:/i);
  const body = (flagsAt >= 0 ? text.slice(0, flagsAt) : text).replace(/^\s*PATIENT-FRIENDLY VERSION:\s*/i, "");
  const flags = flagsAt >= 0 ? text.slice(flagsAt).replace(/^REVIEW FLAGS:\s*/i, "") : "";
  return { body: body.trim(), flags: flags.trim() };
}

/** Remove inline Markdown markers for plain display of a span. */
export function plain(text: string): string {
  return text
    .replace(/^#{1,6}\s+/, "")
    .replace(/^[-*+]\s+/, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/`([^`]+)`/g, "$1");
}

export type DiffPart = { kind: "same" | "add" | "del"; text: string };

/** Word-level diff (LCS) for showing exactly what the Refiner changed. */
export function wordDiff(before: string, after: string): DiffPart[] {
  const a = before.split(/(\s+)/);
  const b = after.split(/(\s+)/);
  const n = a.length;
  const m = b.length;
  const table: Uint32Array[] = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const parts: DiffPart[] = [];
  const push = (kind: DiffPart["kind"], text: string) => {
    const last = parts[parts.length - 1];
    if (last && last.kind === kind) last.text += text;
    else parts.push({ kind, text });
  };
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      push("same", a[i]);
      i++;
      j++;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      push("del", a[i++]);
    } else {
      push("add", b[j++]);
    }
  }
  while (i < n) push("del", a[i++]);
  while (j < m) push("add", b[j++]);
  return parts;
}

/** Turn raw model/ADK errors into something a demo audience can read. */
export function friendlyError(error: string): string {
  if (/\b503\b|UNAVAILABLE|high demand/i.test(error))
    return "The model provider is overloaded right now (503). Try again in a minute, or replay a recorded run.";
  if (/\b500\b|INTERNAL/.test(error))
    return "The model provider had an internal error (500). Try again, or replay a recorded run.";
  if (/\b429\b|RESOURCE_EXHAUSTED|quota/i.test(error)) return "The model API quota is used up (429). Replay a recorded run instead.";
  if (/API key|PERMISSION_DENIED|\b40[13]\b/i.test(error)) return "The server's API key was rejected. Replay a recorded run instead.";
  return error.length > 300 ? `${error.slice(0, 300)}…` : error;
}

export function seconds(start: string | null, end: string | null): number | null {
  if (!start || !end) return null;
  return (Date.parse(end) - Date.parse(start)) / 1000;
}
