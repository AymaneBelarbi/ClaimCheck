import { useState } from "react";
import { askClaim } from "../api";
import type { AskAnswer } from "../types";

export function AskBox({ claimId, token }: { claimId: string; token: string }) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AskAnswer | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      setResult(await askClaim(token, claimId, question.trim()));
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="ask-h">
      <h3 id="ask-h">Ask about this claim</h3>
      <form className="ask" onSubmit={submit}>
        <input
          aria-label="Question"
          value={question}
          maxLength={2000}
          placeholder="e.g. Is the collision covered under this policy version?"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Asking" : "Ask"}
        </button>
      </form>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {result && (
        <div className="answer">
          {result.uncited && (
            <p className="warn" role="alert">
              Uncited: this answer is not backed by a retrieved policy clause. Check it before relying on it.
            </p>
          )}
          <p className="answer-text">{result.answer}</p>
          <p className="muted">
            Cited clauses:{" "}
            {result.citations.length === 0
              ? "none"
              : result.citations.map((c) => (
                  <code key={c} className="chip">
                    {c}
                  </code>
                ))}
          </p>
          <p className="muted">Tools: {result.tools_called.join(", ") || "none"}</p>
        </div>
      )}
    </section>
  );
}
