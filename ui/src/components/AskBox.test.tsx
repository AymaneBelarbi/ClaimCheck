import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AskBox } from "./AskBox";

function stubAsk(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
}

afterEach(() => vi.unstubAllGlobals());

describe("AskBox", () => {
  async function ask() {
    render(<AskBox claimId="c1" token="t" />);
    await userEvent.type(screen.getByRole("textbox", { name: "Question" }), "Is glass covered?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
  }

  it("warns when the answer is uncited", async () => {
    stubAsk({ answer: "Probably.", citations: [], uncited: true, tools_called: ["get_claim"] });
    await ask();
    expect(await screen.findByText(/^Uncited:/)).toBeInTheDocument();
    expect(screen.getByText("Probably.")).toBeInTheDocument();
  });

  it("shows cited clause refs and no warning when the answer is cited", async () => {
    stubAsk({ answer: "Yes.", citations: ["2.1"], uncited: false, tools_called: ["search_policy_clauses"] });
    await ask();
    expect(await screen.findByText("2.1")).toBeInTheDocument();
    expect(screen.queryByText(/^Uncited:/)).not.toBeInTheDocument();
  });

  it("surfaces the API's detail text on failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "the model backend did not answer" }), { status: 502 })));
    await ask();
    expect(await screen.findByText("the model backend did not answer")).toBeInTheDocument();
  });
});
