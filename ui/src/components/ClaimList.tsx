import { CLAIM_STATUSES, type ClaimStatus, type ClaimSummary } from "../types";

export function StatusText({ status }: { status: ClaimStatus }) {
  return <span className={`status status-${status}`}>{status.replace(/_/g, " ")}</span>;
}

interface Props {
  claims: ClaimSummary[];
  selectedId: string | null;
  filter: ClaimStatus | "";
  loading: boolean;
  onFilter: (status: ClaimStatus | "") => void;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}

export function ClaimList({ claims, selectedId, filter, loading, onFilter, onSelect, onRefresh }: Props) {
  return (
    <nav className="claims" aria-label="Claims">
      <div className="claims-bar">
        <label>
          <span className="label">Status</span>
          <select value={filter} onChange={(e) => onFilter(e.target.value as ClaimStatus | "")}>
            <option value="">all</option>
            {CLAIM_STATUSES.map((s) => (
              <option key={s} value={s}>
                {s.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={onRefresh} disabled={loading}>
          Refresh
        </button>
      </div>
      {claims.length === 0 && <p className="muted pad">{loading ? "Loading" : "No claims"}</p>}
      <ul>
        {claims.map((c) => (
          <li key={c.claim_id}>
            <button
              type="button"
              className="claim-row"
              aria-current={c.claim_id === selectedId}
              onClick={() => onSelect(c.claim_id)}
            >
              <span className="mono">{c.packet_id}</span>
              <StatusText status={c.status} />
              <span className="muted">
                {c.claim_type} <span className="num">{c.incident_date}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
