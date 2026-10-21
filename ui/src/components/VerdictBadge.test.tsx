import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Verdict } from "../types";
import { VerdictBadge } from "./VerdictBadge";

describe("VerdictBadge", () => {
  it("spells low_confidence as text", () => {
    render(<VerdictBadge verdict="low_confidence" />);
    expect(screen.getByText("low confidence")).toBeInTheDocument();
  });

  it("uses a distinct class for each verdict", () => {
    const verdicts: Verdict[] = ["found", "missing", "conflicting", "low_confidence"];
    const classes = verdicts.map((verdict) => {
      const { container, unmount } = render(<VerdictBadge verdict={verdict} />);
      const cls = container.firstElementChild?.className ?? "";
      unmount();
      return cls;
    });
    expect(new Set(classes).size).toBe(4);
    for (const cls of classes) expect(cls).toMatch(/verdict-/);
  });
});
