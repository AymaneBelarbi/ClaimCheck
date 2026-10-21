import type { AskAnswer, ClaimStatus, ClaimSummary, Decision, ReviewFile } from "./types";

const BASE: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";
const TOKEN_KEY = "claimcheck.token";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

// FastAPI sends `detail` as a string for HTTPException and as a list of
// {loc, msg} objects for request validation errors.
function detailText(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d as { msg?: string }).msg)
      .filter(Boolean)
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(init.body ? { "Content-Type": "application/json" } : {}),
      },
    });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${BASE}`);
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, detailText(body, `${res.status} ${res.statusText}`));
  return body as T;
}

export const listClaims = (token: string, status: ClaimStatus | "") =>
  request<ClaimSummary[]>(`/claims${status ? `?status=${status}` : ""}`, token);

export const getClaim = (token: string, id: string) =>
  request<ClaimSummary>(`/claims/${id}`, token);

export const getReview = (token: string, id: string) =>
  request<ReviewFile>(`/claims/${id}/review`, token);

export const askClaim = (token: string, id: string, question: string) =>
  request<AskAnswer>(`/claims/${id}/ask`, token, { method: "POST", body: JSON.stringify({ question }) });

export const decideClaim = (token: string, id: string, decision: Decision, reason: string) =>
  request<{ claim_id: string; status: ClaimStatus }>(
    `/claims/${id}/decision`,
    token,
    { method: "POST", body: JSON.stringify({ decision, reason }) },
  );

// The token is read for display only. The API verifies it; nothing here does.
export interface TokenInfo {
  sub: string;
  role: string;
}

export function decodeToken(token: string): TokenInfo | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const b64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const claims = JSON.parse(atob(b64.padEnd(Math.ceil(b64.length / 4) * 4, "="))) as {
      sub?: string;
      role?: string;
    };
    if (!claims.sub || !claims.role) return null;
    return { sub: claims.sub, role: claims.role };
  } catch {
    return null;
  }
}

export function loadToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage can be blocked; the token then lasts until the page is closed.
  }
}
