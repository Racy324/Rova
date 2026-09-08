from __future__ import annotations

from dataclasses import dataclass


RUN_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunManifestRecord:
    """Suite-local link from one EvalResult to its frozen experiment facts."""

    experiment: str
    case_id: str
    profile: str
    repeat_index: int
    fixture_id: str | None
    fixture_sha256: str | None
    eval_run_id: str
    run_id: str | None
    provider_requests: int

    def __post_init__(self) -> None:
        if not self.experiment or not self.case_id or not self.profile or not self.eval_run_id:
            raise ValueError("run manifest requires non-empty experiment, case_id, profile, and eval_run_id")
        if not isinstance(self.repeat_index, int) or isinstance(self.repeat_index, bool) or self.repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")
        if not isinstance(self.provider_requests, int) or isinstance(self.provider_requests, bool) or self.provider_requests < 0:
            raise ValueError("provider_requests must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "experiment": self.experiment,
            "case_id": self.case_id,
            "profile": self.profile,
            "repeat_index": self.repeat_index,
            "fixture_id": self.fixture_id,
            "fixture_sha256": self.fixture_sha256,
            "eval_run_id": self.eval_run_id,
            "run_id": self.run_id,
            "provider_requests": self.provider_requests,
        }
