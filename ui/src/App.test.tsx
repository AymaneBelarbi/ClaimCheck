import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { reviewFixture } from "./fixtures";

const summary = {
  claim_id: reviewFixture.claim.id,
  packet_id: "PKT-007",
  status: "processing",
  claim_type: "collision",
  incident_date: "2026-03-14",
  created_at: "2026-03-14T10:00:00Z",
};

const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

describe("App polling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    sessionStorage.setItem("claimcheck.token", "a.b.c");
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("polls a processing claim every 2 s and loads the review once it changes", async () => {
    let detailCalls = 0;
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/review")) return json(reviewFixture);
      if (url.endsWith(`/claims/${summary.claim_id}`)) {
        detailCalls += 1;
        return json({ ...summary, status: detailCalls < 2 ? "processing" : "needs_review" });
      }
      return json([summary]);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: /PKT-007/ }));
    expect(screen.getByText(/The review loads when processing finishes/)).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(detailCalls).toBe(1);
    expect(fetchMock).not.toHaveBeenCalledWith(expect.stringContaining("/review"), expect.anything());

    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(detailCalls).toBe(2);
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(screen.getByRole("region", { name: "Summary" })).toBeInTheDocument();
  });

  it("stops polling once the API answers 401, 403 or 404", async () => {
    let detailCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.endsWith(`/claims/${summary.claim_id}`)) {
          detailCalls += 1;
          return new Response(JSON.stringify({ detail: "claim not found" }), { status: 404 });
        }
        return json([summary]);
      }),
    );
    render(<App />);
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: /PKT-007/ }));
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(detailCalls).toBe(1);
    expect(screen.getByText("claim not found")).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(detailCalls).toBe(1);
  });
});
