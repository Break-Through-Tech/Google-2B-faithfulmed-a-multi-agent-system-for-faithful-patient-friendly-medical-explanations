import { Fragment, type ReactNode } from "react";

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : <Fragment key={i}>{part}</Fragment>,
  );
}

/** Minimal Markdown (headings, bullets, bold, rules) for agent output. */
export function Markdown({ text, className = "" }: { text: string; className?: string }) {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) blocks.push(<p key={blocks.length}>{inline(paragraph.join(" "))}</p>);
    if (bullets.length)
      blocks.push(
        <ul key={blocks.length}>
          {bullets.map((b, i) => (
            <li key={i}>{inline(b)}</li>
          ))}
        </ul>,
      );
    paragraph = [];
    bullets = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^[-*+]\s+(.*)$/);
    if (!line) flush();
    else if (/^(-{3,}|\*{3,})$/.test(line)) {
      flush();
      blocks.push(<hr key={blocks.length} />);
    } else if (heading) {
      flush();
      blocks.push(<h3 key={blocks.length}>{inline(heading[2])}</h3>);
    } else if (bullet) {
      if (paragraph.length) flush();
      bullets.push(bullet[1]);
    } else {
      if (bullets.length) flush();
      paragraph.push(line);
    }
  }
  flush();
  return <div className={`prose-fm ${className}`}>{blocks}</div>;
}
