"""Wrap OpenJudge's GradingRunner with explicit graders, mappers, and timeouts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any, cast

from openjudge.generator.iterative_rubric.query_rubric_generator import POINTWISE_EVALUATION_TEMPLATE
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderError, GraderRank, GraderScore
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

from eva_agent.adapters.openjudge.catalog import GraderEntry, load_class
from eva_agent.adapters.openjudge.normalize import normalize_results
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome


class _TimeoutGrader(BaseGrader):
    """Apply a per-call timeout while preserving the inner grader's behavior."""

    def __init__(self, inner: BaseGrader, timeout_s: float) -> None:
        super().__init__(name=inner.name, mode=inner.mode, description=inner.description)
        self._inner = inner
        self._timeout_s = timeout_s

    async def _aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank | GraderError:
        try:
            result = await asyncio.wait_for(self._inner.aevaluate(executor=None, **kwargs), timeout=self._timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(f"grader '{self.name}' timed out after {self._timeout_s}s") from exc
        return cast(GraderScore | GraderRank | GraderError, result)


class OpenJudgeRunner:
    def __init__(
        self,
        model: BaseChatModel,
        catalog: Mapping[str, GraderEntry],
        *,
        max_concurrency: int,
        timeout_s: float,
    ) -> None:
        self._model = model
        self._catalog = catalog
        self._max_concurrency = max_concurrency
        self._timeout_s = timeout_s

    def _build_grader(self, spec: GraderSpec) -> BaseGrader:
        if spec.level == "builtin":
            entry = self._catalog.get(spec.builtin_id or "")
            if entry is None:
                raise ValueError(f"unknown grader '{spec.builtin_id}'")
            cls = load_class(entry.klass)
            inner = cls(model=self._model) if entry.needs_model else cls()
            return _TimeoutGrader(inner, self._timeout_s)
        if spec.level == "llm_rubric":
            if not spec.rubric:
                raise ValueError(f"llm_rubric grader '{spec.name}' has no rubric")
            inner = LLMGrader(
                model=self._model,
                name=spec.name,
                template=POINTWISE_EVALUATION_TEMPLATE,
                rubrics=spec.rubric,
                task_description_section="",
                min_score=0,
                max_score=1,
                language="zh",
            )
            return _TimeoutGrader(inner, self._timeout_s)
        raise ValueError(f"grader level '{spec.level}' is not implemented in phase 1")

    @staticmethod
    def _mapper_for(spec: GraderSpec, entry: GraderEntry | None) -> dict[str, str]:
        if spec.input_map:
            return dict(spec.input_map)
        if entry is None:
            return {"query": "query", "response": "response"}
        mapping: dict[str, str] = {}
        for name in entry.inputs:
            if name == "reference_response":
                mapping[name] = "reference"
            elif name == "instruction":
                mapping[name] = "query"
            else:
                mapping[name] = name
        return mapping

    async def run(self, specs: list[GraderSpec], rows: list[dict[str, str]]) -> GraderRunOutput:
        started = time.perf_counter()
        configs: dict[str, Any] = {}
        outcomes: list[GraderSampleOutcome] = []

        for spec in specs:
            if spec.mode != "pointwise":
                outcomes.extend(
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        error="listwise graders are not supported in phase 1",
                    )
                    for index in range(len(rows))
                )
                continue
            entry = self._catalog.get(spec.builtin_id or "") if spec.level == "builtin" else None
            try:
                grader = self._build_grader(spec)
            except Exception as exc:
                message = f"cannot build grader '{spec.name}': {exc}"
                outcomes.extend(
                    GraderSampleOutcome(grader=spec.name, sample_index=index, error=message)
                    for index in range(len(rows))
                )
                continue
            configs[spec.name] = GraderConfig(grader=grader, mapper=self._mapper_for(spec, entry))

        error: str | None = None
        if configs:
            try:
                results = await GradingRunner(
                    grader_configs=configs,
                    max_concurrency=self._max_concurrency,
                    show_progress=False,
                ).arun(rows)
            except Exception as exc:
                error = f"grading runner failed: {exc}"
                handled = {item.grader for item in outcomes}
                outcomes.extend(
                    GraderSampleOutcome(grader=spec.name, sample_index=index, error=error)
                    for spec in specs
                    if spec.name not in handled
                    for index in range(len(rows))
                )
            else:
                normalized = normalize_results(
                    results,
                    [spec for spec in specs if spec.name in configs],
                    self._catalog,
                    len(rows),
                )
                outcomes.extend(normalized)
        elif not outcomes:
            error = "no runnable graders"

        duration_ms = int((time.perf_counter() - started) * 1000)
        return GraderRunOutput(outcomes=outcomes, duration_ms=duration_ms, error=error)
