import type { ServerEvent } from "./types";

/** Read a Server-Sent Events response (GET or POST) and hand each event to `onEvent`. */
export async function streamEvents(
  url: string,
  init: RequestInit,
  onEvent: (event: ServerEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(url, { ...init, signal });
  if (!response.ok || !response.body) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep the status message */
    }
    throw new Error(message);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (data) onEvent(JSON.parse(data) as ServerEvent);
    }
  }
}
