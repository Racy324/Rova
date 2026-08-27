from __future__ import annotations

from typing import Protocol

from .models import CheckResult, EvalCase, EvalExecution


class Evaluator(Protocol):
    async def evaluate(self, case: EvalCase, execution: EvalExecution) -> CheckResult: ...
