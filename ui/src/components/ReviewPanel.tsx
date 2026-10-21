import { Fragment, useState } from "react";
import { decideClaim } from "../api";
import type { Decision, Evidence, ReviewDocument, ReviewField, ReviewFile, Verdict } from "../types";
import { AskBox } from "./AskBox";
import { StatusText } from "./ClaimList";
import { VerdictBadge } from "./VerdictBadge";

const VERDICTS: Verdict[] = ["found", "missing", "conflicting", "low_confidence"];
const pct = (c: number) => `${Math.round(c * 100)}%`;

function Counts({ counts }: { counts: Record<Verdict, number> }) {
  return (
    <>
      {VERDICTS.map((v) => (
        <span key={v} className="count">
          <VerdictBadge verdict={v} /> <span className="num">{counts[v]}</span>
        </span>
      ))}
    </>
  );
}

// line_items is stored as canonical JSON, [{"amount", "description"}]; anything else shows as text.
function LineItems({ raw }: { raw: string | null }) {
  let items: { amount: unknown; description: unknown }[] | null = null;
  try {
    const parsed: unknown = JSON.parse(raw ?? "");
    if (Array.isArray(parsed)) items = parsed;
  } catch {
    // not JSON: fall through to the raw string
  }
  if (!items) return <>{raw ?? <span className="muted">none</span>}</>;
  return (
    <ul className="line-items">
      {items.map((it, i) => (
        <li key={i}>
          <span>{String(it?.description ?? "")}</span>
          <span className="num">{String(it?.amount ?? "")}</span>
        </li>
      ))}
    </ul>
  );
}

function EvidenceItems({ items, names }: { items: Evidence[]; names: Map<string, string> }) {
  if (items.length === 0) return <p className="muted">No evidence recorded.</p>;
  return (
    <ul className="evidence">
      {items.map((ev, i) => (
        <li key={i}>
          <span className="mono">{names.get(ev.document_id) ?? ev.document_id}</span>
          {ev.page !== null && <span className="num"> p.{ev.page}</span>}
          {ev.span && (
            <span className="num muted">
              {" "}
              chars {ev.span[0]}-{ev.span[1]}
            </span>
          )}
          {ev.text && <blockquote>{ev.text}</blockquote>}
        </li>
      ))}
    </ul>
  );
}

// A row plus an optional detail row. The toggle is a real button so it works from the keyboard.
function ExpandableRow({ label, children, evidence, names }: { label: string; children: React.ReactNode; evidence: Evidence[]; names: Map<string, string> }) {
  const [open, setOpen] = useState(false);
  return (
    <Fragment>
      <tr>
        {children}
        <td>
          <button type="button" className="link" aria-label={`Evidence for ${label}`} aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "Hide" : "Evidence"}
          </button>
        </td>
      </tr>
      {open && (
        <tr className="detail">
          <td colSpan={5}>
            <EvidenceItems items={evidence} names={names} />
          </td>
        </tr>
      )}
    </Fragment>
  );
}

function DocumentBlock({ doc, fields, names }: { doc: ReviewDocument; fields: ReviewField[]; names: Map<string, string> }) {
  return (
    <div className="doc">
      <h4>
        <span className="mono">{doc.filename}</span>{" "}
        <span className="muted">{doc.doc_type ?? "not classified"}</span>
      </h4>
      <p className="muted">
        {doc.classification_confidence !== null && (
          <>
            classified {pct(doc.classification_confidence)}
            {doc.classification_method ? ` (${doc.classification_method})` : ""}
            {" · "}
          </>
        )}
        {doc.ocr_mean_confidence !== null && <>OCR {pct(doc.ocr_mean_confidence)}</>}
        {doc.low_confidence && <strong className="flag"> low-confidence scan</strong>}
      </p>
      {doc.error && <p className="error">{doc.error}</p>}
      {fields.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Field</th>
              <th>Value</th>
              <th>Confidence</th>
              <th>Verdict</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => (
              <ExpandableRow key={f.field_name} label={f.field_name} names={names} evidence={f.evidence ? [f.evidence] : []}>
                <td className="mono">{f.field_name}</td>
                <td>{f.field_name === "line_items" ? <LineItems raw={f.value} /> : (f.value ?? <span className="muted">none</span>)}</td>
                <td className="num">{pct(f.confidence)}</td>
                <td>
                  <VerdictBadge verdict={f.verdict} />
                </td>
              </ExpandableRow>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function DecisionBar({ review, token, onDecided }: { review: ReviewFile; token: string; onDecided: () => void }) {
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // A claim with blocking rules needs the adjuster's own reason even to approve.
  const reasonRequired = !review.summary.ready_for_decision;
  const hasReason = reason.trim() !== "";

  async function decide(decision: Decision) {
    setBusy(true);
    setError("");
    try {
      // The API wants a reason on every decision; approval falls back to a fixed one.
      await decideClaim(token, review.claim.id, decision, reason.trim() || "approved by adjuster");
      onDecided();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="decision-h">
      <h3 id="decision-h">Decision</h3>
      <label>
        <span className="label">
          {reasonRequired
            ? "Reason (required: rules are blocking this claim)"
            : "Reason (required to reject; approval without one is recorded as \"approved by adjuster\")"}
        </span>
        <textarea rows={2} value={reason} maxLength={2000} onChange={(e) => setReason(e.target.value)} />
      </label>
      <div className="actions">
        <button type="button" disabled={busy || (reasonRequired && !hasReason)} onClick={() => decide("approved_for_review")}>
          Approve for review
        </button>
        <button type="button" className="danger" disabled={busy || !hasReason} onClick={() => decide("rejected_incomplete")}>
          Reject as incomplete
        </button>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

export function ReviewPanel({ review, token, onDecided }: { review: ReviewFile; token: string; onDecided: () => void }) {
  const { claim, summary, policy_version: pv } = review;
  const names = new Map(review.documents.map((d) => [d.id, d.filename]));
  return (
    <article className="review">
      <header>
        <h2 className="mono">{claim.packet_id}</h2>
        <StatusText status={claim.status} />
        <span className="muted">
          {claim.claim_type} · incident <span className="num">{claim.incident_date}</span>
          {pv && (
            <>
              {" "}
              · policy v{pv.version_no} <span className="num">{pv.effective_from}</span> to{" "}
              <span className="num">{pv.effective_to ?? "open"}</span>
            </>
          )}
        </span>
      </header>

      <section aria-labelledby="summary-h">
        <h3 id="summary-h">Summary</h3>
        <p>
          <strong className={summary.ready_for_decision ? "ok" : "flag"}>
            {summary.ready_for_decision ? "Ready for decision" : "Not ready for decision"}
          </strong>
        </p>
        <dl className="counts">
          <dt>Rules</dt>
          <dd>
            <Counts counts={summary.verdict_counts} />
          </dd>
          <dt>Fields</dt>
          <dd>
            <Counts counts={summary.field_verdict_counts} />
          </dd>
          <dt>Blocking</dt>
          <dd>
            {summary.blocking.length === 0
              ? "none"
              : summary.blocking.map((id) => (
                  <code key={id} className="chip">
                    {id}
                  </code>
                ))}
          </dd>
        </dl>
      </section>

      <section aria-labelledby="rules-h">
        <h3 id="rules-h">Rules</h3>
        <table>
          <thead>
            <tr>
              <th>Rule</th>
              <th>Description</th>
              <th>Verdict</th>
              <th>Message</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {review.rules.map((r) => (
              <ExpandableRow key={r.rule_id} label={r.rule_id} names={names} evidence={r.evidence}>
                <td className="mono">{r.rule_id}</td>
                <td>{r.description}</td>
                <td>
                  <VerdictBadge verdict={r.verdict} />
                </td>
                <td>{r.message}</td>
              </ExpandableRow>
            ))}
          </tbody>
        </table>
      </section>

      <section aria-labelledby="docs-h">
        <h3 id="docs-h">Documents</h3>
        {review.documents.map((d) => (
          <DocumentBlock key={d.id} doc={d} names={names} fields={review.fields.filter((f) => f.document_id === d.id)} />
        ))}
      </section>

      <AskBox key={claim.id} claimId={claim.id} token={token} />
      {claim.status === "needs_review" && <DecisionBar key={claim.id} review={review} token={token} onDecided={onDecided} />}
    </article>
  );
}
