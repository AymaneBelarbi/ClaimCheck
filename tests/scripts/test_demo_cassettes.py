"""The demo runs on LLM_BACKEND=replay, so every call its default path can make must be recorded.

The classification fallback is included for every document even though the embeddings decide most
of them on the recording machine: another Tesseract build reads the scans differently and can
push a document under the margin, and a miss there fails the demo's job.
"""

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from evals.dataset.types import GroundTruth
from gateway.replay import cassette_filename

ROOT = Path(__file__).parents[2]
CASSETTES = ROOT / "evals" / "cassettes"
DATA = ROOT / "evals" / "data"


def _demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts" / "demo.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _keys() -> list[str]:
    demo = _demo()
    packet: str = demo.DEFAULT_PACKET
    truth_file = DATA / "packets" / packet / "ground_truth.json"
    if not truth_file.exists():
        pytest.skip("evals/data not generated (make dataset)")
    truth = GroundTruth.model_validate_json(truth_file.read_text(encoding="utf-8"))
    digest = hashlib.sha256(demo.DEFAULT_QUESTION.encode("utf-8")).hexdigest()[:12]
    keys = [f"classify:{packet}:{d.filename}" for d in truth.documents]
    keys += [f"extract:{packet}:{d.filename}" for d in truth.documents]
    keys.append(f"agent:{packet}:{digest}:turn1")
    return keys


def test_every_call_of_the_default_demo_is_recorded() -> None:
    missing = [key for key in _keys() if not (CASSETTES / cassette_filename(key)).exists()]

    assert missing == []
