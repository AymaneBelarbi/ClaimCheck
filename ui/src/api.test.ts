import { afterEach, describe, expect, it, vi } from "vitest";
import { decodeToken, getReview } from "./api";

const b64 = (o: object) => btoa(JSON.stringify(o)).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
const jwt = (claims: object) => `${b64({ alg: "HS256" })}.${b64(claims)}.sig`;

afterEach(() => vi.unstubAllGlobals());

describe("decodeToken", () => {
  it("reads sub and role without verifying", () => {
    expect(decodeToken(jwt({ sub: "adj-1", role: "adjuster" }))).toEqual({ sub: "adj-1", role: "adjuster" });
  });

  it("rejects non-JWTs", () => {
    expect(decodeToken("not-a-token")).toBeNull();
  });
});

describe("request errors", () => {
  it("surfaces a string detail", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "claim not found" }), { status: 404 })));
    await expect(getReview("t", "x")).rejects.toMatchObject({ message: "claim not found" });
  });

  it("joins validation messages from a 422", async () => {
    const detail = [{ msg: "String should have at least 1 character" }];
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail }), { status: 422 })));
    await expect(getReview("t", "x")).rejects.toThrow("String should have at least 1 character");
  });

  it("reports an unreachable API", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("failed"))));
    await expect(getReview("t", "x")).rejects.toThrow("Cannot reach the API");
  });
});
