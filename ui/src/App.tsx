import { useCallback, useEffect, useState } from "react";
import { ApiError, decodeToken, getClaim, getReview, listClaims, loadToken, saveToken } from "./api";
import { ClaimList } from "./components/ClaimList";
import { ReviewPanel } from "./components/ReviewPanel";
import type { ClaimStatus, ClaimSummary, ReviewFile } from "./types";

const POLL_MS = 2000;
const IN_FLIGHT: ClaimStatus[] = ["received", "processing"];
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function App() {
  const [token, setToken] = useState(loadToken);
  const [filter, setFilter] = useState<ClaimStatus | "">("");
  const [claims, setClaims] = useState<ClaimSummary[]>([]);
  const [listing, setListing] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [review, setReview] = useState<ReviewFile | null>(null);
  const [error, setError] = useState("");
  // Bumped after a decision so the review is fetched again.
  const [reviewTick, setReviewTick] = useState(0);

  const who = decodeToken(token);
  const selected = claims.find((c) => c.claim_id === selectedId) ?? null;
  const selectedStatus = selected?.status ?? null;
  const inFlight = selectedStatus !== null && IN_FLIGHT.includes(selectedStatus);

  const refreshList = useCallback(async () => {
    if (!token) {
      setClaims([]);
      return;
    }
    setListing(true);
    try {
      setClaims(await listClaims(token, filter));
      setError("");
    } catch (e) {
      setClaims([]);
      setError(message(e));
    } finally {
      setListing(false);
    }
  }, [token, filter]);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  // A claim still in the pipeline has no review file yet; watch it until its status moves.
  useEffect(() => {
    if (!token || !selectedId || !inFlight) return;
    const timer = setInterval(async () => {
      try {
        const claim = await getClaim(token, selectedId);
        setClaims((all) => all.map((c) => (c.claim_id === selectedId ? { ...c, status: claim.status } : c)));
      } catch (e) {
        setError(message(e));
        // A rejected token or a missing claim will not fix itself; a network blip might.
        if (e instanceof ApiError && [401, 403, 404].includes(e.status)) clearInterval(timer);
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [token, selectedId, inFlight]);

  useEffect(() => {
    setReview(null);
    if (!token || !selectedId || selectedStatus === null || inFlight) return;
    let stale = false;
    getReview(token, selectedId)
      .then((r) => {
        if (!stale) {
          setReview(r);
          setError("");
        }
      })
      .catch((e) => {
        if (!stale) setError(message(e));
      });
    return () => {
      stale = true;
    };
  }, [token, selectedId, selectedStatus, inFlight, reviewTick]);

  function changeToken(value: string) {
    const trimmed = value.trim();
    setToken(trimmed);
    saveToken(trimmed);
  }

  function decided() {
    setReviewTick((n) => n + 1);
    void refreshList();
  }

  return (
    <div className="app">
      <header className="top">
        <h1>ClaimCheck</h1>
        <label className="token">
          <span className="label">API token</span>
          <input
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={token}
            placeholder="paste a JWT (make token)"
            onChange={(e) => changeToken(e.target.value)}
          />
        </label>
        <span className="who" aria-live="polite">
          {token === "" ? (
            "no token"
          ) : who ? (
            <>
              <span className="mono">{who.sub}</span> · {who.role}
            </>
          ) : (
            <strong className="flag">not a JWT</strong>
          )}
        </span>
      </header>
      <ClaimList
        claims={claims}
        selectedId={selectedId}
        filter={filter}
        loading={listing}
        onFilter={setFilter}
        onSelect={setSelectedId}
        onRefresh={() => void refreshList()}
      />
      <main>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {!token && <p className="muted pad">Paste an adjuster token above to load claims.</p>}
        {token && !selected && <p className="muted pad">Select a claim.</p>}
        {selected && inFlight && (
          <p className="muted pad" aria-live="polite">
            <span className="mono">{selected.packet_id}</span> is {selected.status}. The review loads when
            processing finishes.
          </p>
        )}
        {review && <ReviewPanel review={review} token={token} onDecided={decided} />}
      </main>
    </div>
  );
}
