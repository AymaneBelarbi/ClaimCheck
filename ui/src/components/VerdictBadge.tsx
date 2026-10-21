import type { Verdict } from "../types";

// The text label is always rendered; colour only reinforces it.
export function verdictLabel(verdict: Verdict): string {
  return verdict.replace("_", " ");
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return <span className={`badge verdict-${verdict.replace("_", "-")}`}>{verdictLabel(verdict)}</span>;
}
