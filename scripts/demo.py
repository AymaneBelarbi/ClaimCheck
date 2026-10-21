"""One claim through the running stack, end to end.

    make up && make demo          (or: uv run python -m scripts.demo [--packet PKT-050])

Waits for the API, seeds the policies, uploads one generated packet as an adjuster, waits for
the worker to finish it, prints the review summary and asks the agent one question about the
policy. It talks to the API over HTTP like any client; only the seed and the token are made
locally (the seed needs the owner login, which the API does not have).

The stack answers model calls with whatever LLM_BACKEND it was started with. The compose
default is `replay`, which serves the recorded responses in evals/cassettes, so the default
packet and question need no model. With `LLM_BACKEND=ollama` or `anthropic` the same script
runs against a real model.

The packet id is the upload's Idempotency-Key. Running the demo again finds the claim filed the
first time and shows it again instead of filing a second one.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from claimcheck.api.tokens import compose_secret, issue_token
from claimcheck.core.types import Role
from evals.dataset.types import GroundTruth
from evals.seed import main as seed

DEFAULT_PACKET = "PKT-050"
DEFAULT_QUESTION = "What is the most the policy pays for broken glass on this claim?"
FINISHED = {"needs_review", "approved_for_review", "rejected_incomplete"}

_MIME = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one claim through the running stack.")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data", type=Path, default=Path("evals/data"))
    parser.add_argument("--packet", default=DEFAULT_PACKET)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait per step")
    args = parser.parse_args(argv)

    folder = args.data / "packets" / args.packet
    if not (folder / "ground_truth.json").exists():
        print(f"{folder} not found; run `make dataset` first", file=sys.stderr)
        return 2
    truth = GroundTruth.model_validate_json((folder / "ground_truth.json").read_text("utf-8"))

    with httpx.Client(base_url=args.api, timeout=60.0) as api:
        step("waiting for the API")
        wait_for_health(api, args.timeout)

        step("seeding policies")
        if seed(["--data", str(args.data)]) != 0:
            return 1

        api.headers["authorization"] = f"Bearer {adjuster_token()}"
        step(
            f"uploading {args.packet}: {truth.claim_type.value} claim on {truth.policy_number}, "
            f"incident {truth.incident_date}, {len(truth.documents)} documents"
        )
        claim_id = upload(api, folder, truth)

        step("waiting for the worker")
        status = wait_for_status(api, claim_id, args.timeout)
        if status not in FINISHED:
            print(f"claim {claim_id} ended {status}; see `docker compose logs worker`")
            return 1

        review = api.get(f"/claims/{claim_id}/review")
        review.raise_for_status()
        print_review(review.json())

        step(f"asking: {args.question}")
        # A local model can take minutes over a few tool-calling turns.
        answer = api.post(
            f"/claims/{claim_id}/ask", json={"question": args.question}, timeout=args.timeout
        )
        if answer.status_code != 200:
            print(f"/ask answered {answer.status_code}: {answer.text}")
            return 1
        print_answer(answer.json())
    return 0


def step(text: str) -> None:
    print(f"\n== {text}", flush=True)


def wait_for_health(api: httpx.Client, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if api.get("/health").status_code == 200:
                print("API is up")
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"no healthy API at {api.base_url} after {timeout:.0f} s")
        time.sleep(2)


def adjuster_token() -> str:
    # Signed with the key the stack checks: JWT_SECRET, or compose's default when it is unset.
    return issue_token("adj-demo", Role.ADJUSTER, secret=compose_secret())


def upload(api: httpx.Client, folder: Path, truth: GroundTruth) -> str:
    files = [
        ("files", (d.filename, (folder / d.filename).read_bytes(), _MIME[Path(d.filename).suffix]))
        for d in truth.documents
    ]
    response = api.post(
        "/claims",
        data={
            "policy_number": truth.policy_number,
            "claim_type": truth.claim_type.value,
            "incident_date": truth.incident_date.isoformat(),
        },
        files=files,
        headers={"Idempotency-Key": truth.packet_id},
    )
    if response.status_code not in (200, 202):
        raise SystemExit(f"upload answered {response.status_code}: {response.text}")
    body: dict[str, Any] = response.json()
    if response.status_code == 200:
        print(f"already filed as claim {body['claim_id']} ({body['status']})")
    else:
        print(f"claim {body['claim_id']} queued")
    return str(body["claim_id"])


def wait_for_status(api: httpx.Client, claim_id: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    last = ""
    while True:
        response = api.get(f"/claims/{claim_id}")
        response.raise_for_status()
        status = str(response.json()["status"])
        if status != last:
            print(status)
            last = status
        if status in FINISHED or status == "failed":
            return status
        if time.monotonic() > deadline:
            return status
        time.sleep(1)


def print_review(review: dict[str, Any]) -> None:
    summary = review["summary"]
    version = review["policy_version"]
    step("review file")
    if version:
        print(f"policy version in force: v{version['version_no']} from {version['effective_from']}")
    else:
        print("no policy version in force on the incident date")
    for doc in review["documents"]:
        flag = "  low OCR confidence" if doc["low_confidence"] else ""
        print(f"  {doc['filename']:<12} {doc['doc_type'] or '-':<16}{flag}")
    print("rule verdicts: " + _counts(summary["verdict_counts"]))
    print("field verdicts: " + _counts(summary["field_verdict_counts"]))
    rules = {r["rule_id"]: r for r in review["rules"]}
    if summary["blocking"]:
        print("blocking rules:")
        for rule_id in summary["blocking"]:
            rule = rules[rule_id]
            print(f"  {rule_id} ({rule['verdict']}): {rule['message']}")
    else:
        print("no blocking rules; ready for a decision")


def print_answer(answer: dict[str, Any]) -> None:
    print(f"answer: {answer['answer']}")
    print(f"citations: {', '.join(answer['citations']) or 'none'}")
    if answer["uncited"]:
        print("(marked uncited: the answer does not rest on a retrieved clause)")
    print(f"tools called: {', '.join(answer['tools_called'])}")


def _counts(counts: dict[str, int]) -> str:
    return json.dumps({k: v for k, v in counts.items() if v})


if __name__ == "__main__":
    sys.exit(main())
