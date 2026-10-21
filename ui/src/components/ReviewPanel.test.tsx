import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { reviewFixture } from "../fixtures";
import type { ReviewFile } from "../types";
import { ReviewPanel } from "./ReviewPanel";

describe("ReviewPanel", () => {
  it("lists blocking rules in the summary and shows each rule's verdict label", () => {
    render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    const summary = screen.getByRole("region", { name: "Summary" });
    expect(within(summary).getByText("Not ready for decision")).toBeInTheDocument();
    expect(within(summary).getByText("name_match")).toBeInTheDocument();
    expect(within(summary).getByText("required_docs")).toBeInTheDocument();

    const rules = screen.getByRole("region", { name: "Rules" });
    expect(within(rules).getByText("conflicting")).toBeInTheDocument();
    expect(within(rules).getByText("missing")).toBeInTheDocument();
  });

  it("shows field confidence as a percentage and reveals evidence on demand", async () => {
    render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    expect(screen.getByText("64%")).toBeInTheDocument();
    expect(screen.queryByText("Total 1840.00")).not.toBeInTheDocument();
    const docs = screen.getByRole("region", { name: "Documents" });
    await userEvent.click(within(docs).getByRole("button", { name: "Evidence for total_amount" }));
    expect(screen.getByText("Total 1840.00")).toBeInTheDocument();
  });

  it("offers a decision only while the claim needs review", () => {
    const { rerender } = render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    const decided = { ...reviewFixture, claim: { ...reviewFixture.claim, status: "approved_for_review" as const } };
    rerender(<ReviewPanel review={decided} token="t" onDecided={() => {}} />);
    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
  });

  it("keeps reject disabled until a reason is typed", async () => {
    render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    const reject = screen.getByRole("button", { name: /reject/i });
    expect(reject).toBeDisabled();
    await userEvent.type(screen.getByRole("textbox", { name: /reason/i }), "no incident report");
    expect(reject).toBeEnabled();
  });

  it("names each evidence button after its rule or field", () => {
    render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    expect(screen.getByRole("button", { name: "Evidence for name_match" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Evidence for total_amount" })).toBeInTheDocument();
  });

  it("renders line_items as a description/amount list and falls back to the raw string", () => {
    const withItems = (value: string): ReviewFile => ({
      ...reviewFixture,
      fields: [{ ...reviewFixture.fields[0]!, field_name: "line_items", value }],
    });
    const json = '[{"amount":"120.00","description":"Windscreen"},{"amount":"15.50","description":"Fitting"}]';
    const { rerender } = render(<ReviewPanel review={withItems(json)} token="t" onDecided={() => {}} />);
    const items = screen.getAllByRole("listitem").filter((li) => li.textContent?.includes("Windscreen"));
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent("Windscreen120.00");
    expect(screen.getByText("15.50")).toBeInTheDocument();
    rerender(<ReviewPanel review={withItems("Windscreen 120.00; Fitting 15.50")} token="t" onDecided={() => {}} />);
    expect(screen.getByText("Windscreen 120.00; Fitting 15.50")).toBeInTheDocument();
  });
});

const ready: ReviewFile = {
  ...reviewFixture,
  summary: { ...reviewFixture.summary, blocking: [], ready_for_decision: true },
};

describe("decision", () => {
  afterEach(() => vi.unstubAllGlobals());

  function stubDecision() {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ claim_id: "x", status: "approved_for_review" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }
  const sent = (fetchMock: ReturnType<typeof stubDecision>) => {
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    return { url, body: JSON.parse(init.body as string) as unknown };
  };

  it("requires a reason to approve when rules are blocking", async () => {
    render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    const approve = screen.getByRole("button", { name: /approve/i });
    expect(approve).toBeDisabled();
    expect(screen.getByText(/required: rules are blocking/)).toBeInTheDocument();
    await userEvent.type(screen.getByRole("textbox", { name: /reason/i }), "checked paper copy");
    expect(approve).toBeEnabled();
  });

  it("approves a ready claim without a reason, recording the default one", async () => {
    const fetchMock = stubDecision();
    const onDecided = vi.fn();
    render(<ReviewPanel review={ready} token="t" onDecided={onDecided} />);
    expect(screen.getByText(/recorded as "approved by adjuster"/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /approve/i }));
    expect(sent(fetchMock)).toEqual({
      url: expect.stringContaining(`/claims/${ready.claim.id}/decision`),
      body: { decision: "approved_for_review", reason: "approved by adjuster" },
    });
    expect(onDecided).toHaveBeenCalled();
  });

  it("rejects with the trimmed reason", async () => {
    const fetchMock = stubDecision();
    render(<ReviewPanel review={reviewFixture} token="t" onDecided={() => {}} />);
    await userEvent.type(screen.getByRole("textbox", { name: /reason/i }), "  no incident report  ");
    await userEvent.click(screen.getByRole("button", { name: /reject/i }));
    expect(sent(fetchMock).body).toEqual({ decision: "rejected_incomplete", reason: "no incident report" });
  });
});
