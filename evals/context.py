"""What a stage runner receives, and the runner type."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from claimcheck.core.settings import Settings
from claimcheck.gateway.base import LLMGateway
from claimcheck.gateway.factory import get_gateway
from claimcheck.gateway.replay import ReplayGateway
from claimcheck.gateway.tracing import UsageLedger, unwrap
from claimcheck.pipeline.extract import FieldResult
from claimcheck.rag.embedder import Embedder, FastEmbedEmbedder
from evals.cost import summarize_cost
from evals.report import StageResult


class StageUnavailable(Exception):
    """A stage cannot run because something it depends on is not set up; the message says what."""


class EvalContext:
    """What a stage runner needs. The gateway and embedder are built on first use, so a stage
    that needs neither (validation, say) works without Ollama or the embedding model."""

    def __init__(
        self,
        data_dir: Path,
        settings: Settings,
        *,
        gateway: LLMGateway | None = None,
        embedder: Embedder | None = None,
        record_all_fallbacks: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.settings = settings
        # Recording only: see evals.run.
        self.record_all_fallbacks = record_all_fallbacks
        self._gateway = gateway
        self._embedder = embedder
        # Filled by the stages that process whole packets; see evals.cost.
        self.ledger = UsageLedger()
        # The extraction stage's output, packet id -> filename -> fields, for the validation
        # stage's end-to-end score when both run in one invocation.
        self.extracted: dict[str, dict[str, list[FieldResult]]] = {}

    @property
    def gateway(self) -> LLMGateway:
        if self._gateway is None:
            self._gateway = get_gateway(self.settings)
        return self._gateway

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedEmbedder()
        return self._embedder

    def run_info(self) -> dict[str, Any]:
        """Run metadata for the report; models are only known for what the run actually used."""
        info: dict[str, Any] = {
            "llm_backend": self.settings.llm_backend,
            "llm_model": self._gateway.model if self._gateway else None,
            "embedding_model": self._embedder.name if self._embedder else None,
        }
        if self._gateway is not None:
            replay = unwrap(self._gateway)
            if isinstance(replay, ReplayGateway):
                info["replay_drift_count"] = replay.drift_count
        return info

    def cost_summary(self) -> dict[str, Any] | None:
        """The report's cost section: per-packet cost and latency of the packets this run
        processed, or None when it processed none."""
        model = self._gateway.model if self._gateway else None
        return summarize_cost(self.ledger, backend=self.settings.llm_backend, model=model)


StageRunner = Callable[[str, EvalContext], StageResult]
