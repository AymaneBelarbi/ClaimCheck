"""Generate the synthetic evaluation dataset.

    python -m evals.generate_dataset --out evals/data --seed 20261020

Writes policy markdown and `policies/index.json`, 50 claim packets with `ground_truth.json`,
`splits.json` (35 dev / 15 holdout), and the clause-question set. The same seed gives
byte-identical output, so `evals/data/` is not committed; `clause_questions.jsonl` is, because
the retrieval eval reads it without generating anything.

Each part draws from its own random stream derived from the seed, so changing how packets are
built does not reshuffle the clause questions, and the other way round.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from evals.dataset.policies import POLICIES, render_markdown
from evals.dataset.questions import build_questions
from evals.dataset.render import IdCardDoc, render_id_card, render_pdf, render_unreadable
from evals.dataset.scenarios import Packet, build_packets, split
from evals.dataset.types import PolicyEntry, PolicyIndex, PolicyVersionEntry

DEFAULT_QUESTIONS = Path(__file__).parent / "clause_questions.jsonl"
HOLDOUT = 15


def _stream(seed: int, name: str) -> random.Random:
    return random.Random(f"{seed}/{name}")


def write_policies(out: Path) -> None:
    root = out / "policies"
    entries = []
    for policy in POLICIES:
        folder = root / policy.policy_number
        folder.mkdir(parents=True)
        versions = []
        for v in policy.versions:
            rel = f"{policy.policy_number}/v{v.version_no}.md"
            (root / rel).write_text(render_markdown(policy, v))
            versions.append(
                PolicyVersionEntry(
                    version_no=v.version_no,
                    effective_from=v.effective_from,
                    effective_to=v.effective_to,
                    file=rel,
                )
            )
        entries.append(
            PolicyEntry(
                policy_number=policy.policy_number, holder_id=policy.holder_id, versions=versions
            )
        )
    (root / "index.json").write_text(PolicyIndex(policies=entries).model_dump_json(indent=2) + "\n")


def write_packet(out: Path, packet: Packet, rng: random.Random) -> None:
    folder = out / "packets" / packet.packet_id
    folder.mkdir(parents=True)
    for f in packet.files:
        path = folder / f.filename
        angle = rng.uniform(-2.0, 2.0)
        noise_seed = rng.getrandbits(32)
        if f.unreadable:
            render_unreadable(f.doc, path, angle, noise_seed)
        elif isinstance(f.doc, IdCardDoc):
            render_id_card(f.doc, path, angle, noise_seed)
        else:
            render_pdf(f.doc, path)
    (folder / "ground_truth.json").write_text(packet.truth.model_dump_json(indent=2) + "\n")


def generate(out: Path, seed: int, questions_path: Path = DEFAULT_QUESTIONS) -> None:
    # Only the parts this script owns are replaced; anything else under `out` is left alone.
    for part in ("policies", "packets"):
        shutil.rmtree(out / part, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    write_policies(out)

    packets = build_packets(_stream(seed, "packets"))
    scans = _stream(seed, "scans")
    for packet in packets:
        write_packet(out, packet, scans)

    splits = split(packets, _stream(seed, "splits"), HOLDOUT)
    (out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")

    questions = build_questions(_stream(seed, "questions"))
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.write_text("".join(q.model_dump_json() + "\n" for q in questions))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=Path("evals/data"))
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--questions-out", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args(argv)
    generate(args.out, args.seed, args.questions_out)
    print(f"wrote dataset to {args.out} and clause questions to {args.questions_out}")


if __name__ == "__main__":
    main()
