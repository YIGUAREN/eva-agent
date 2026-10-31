from decimal import Decimal
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from eva_agent.budget.estimator import CostEstimator
from eva_agent.config.settings import PricingSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import RunStatus
from eva_agent.pricing import PricingTable
from eva_agent.runtime.manager import RunManager
from eva_agent.worker.runner import RunWorker
from tests.support.api import csrf_headers, login
from tests.support.runtime import FakePlanner, FakeRunner, dependencies, fake_config, paste_input, sample_plan
from tests.support.server import build_test_app

ROWS = [{"query": "q", "response": "a", "context": "", "reference": ""}]


def _config(tmp_path: Path):
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "currency: USD\n"
        "models:\n"
        "  - provider: openai-compatible\n"
        "    model: fake\n"
        "    input_per_million: 0.0\n"
        "    output_per_million: 1.0\n",
        encoding="utf-8",
    )
    return fake_config(tmp_path).model_copy(update={"pricing": PricingSettings(path=str(pricing))})


async def _client(tmp_path: Path, config):
    app, state = await build_test_app(tmp_path, config=config)
    http = AsyncClient(transport=ASGITransport(app=app), base_url="https://test")
    return http, state


async def _start_and_approve(manager, worker, budget: Budget) -> str:
    run_id = await manager.create_run(paste_input(), budget)
    await worker.commands.enqueue(run_id, "start")
    assert await worker.tick() is True
    await worker.commands.enqueue(run_id, "approve", {"approved": True})
    assert await worker.tick() is True
    return run_id


async def test_summarize_needs_confirmation_then_completes_with_an_overrun(tmp_path) -> None:
    config = _config(tmp_path)
    plan = sample_plan(sample_size=1)
    quality = next(item for item in plan.dimensions if item.dimension is Dimension.QUALITY)
    estimator = CostEstimator(PricingTable.load(Path(config.pricing.path)))
    first = estimator.estimate_grader(config.models.judge, spec=quality.graders[0], rows=ROWS)
    second = estimator.estimate_grader(config.models.judge, spec=quality.graders[1], rows=ROWS)
    runner = FakeRunner()
    deps = dependencies(tmp_path, planner=FakePlanner(plan=plan), runner=runner)
    manager = RunManager(config, dependencies=deps, enforce_budget=True)
    await manager.setup()
    worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir)
    # The fake planner does not reserve; the first grader fits, the second does not.
    budget = Budget(
        max_cost_usd=float(first + second / 2),
        max_wall_clock_s=60.0,
        max_target_requests=1,
    )
    run_id = await _start_and_approve(manager, worker, budget)

    assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET
    http, state = await _client(tmp_path, config)
    async with http:
        csrf = await login(http)
        detail = (await http.get(f"/api/runs/{run_id}")).json()
        assert detail["budget_gate"]["truncated_dimensions"] == ["quality"]
        blocked = await http.post(
            f"/api/runs/{run_id}/budget/decision",
            json={"action": "summarize"},
            headers=csrf_headers(csrf),
        )
        assert blocked.status_code == 409
        assert set(blocked.json()["error"]["details"]) == {"finalize_estimate", "remaining"}
        confirmed = await http.post(
            f"/api/runs/{run_id}/budget/decision",
            json={"action": "summarize", "confirm_overrun": True},
            headers=csrf_headers(csrf),
        )
        assert confirmed.status_code == 202

    assert await worker.tick() is True
    assert (await manager.status(run_id)).status is RunStatus.COMPLETED
    report = await manager.report(run_id)
    assert report["version"] == 4
    assert report["finalize_overrun"] is True
    assert report["truncated_dimensions"] == ["quality"]
    assert report["usage"]["budget_enforced"] is True
    assert report["dimensions"][0]["status"] == "partial"
    await manager.close()
    await state.engine.dispose()


async def test_extend_reruns_only_fresh_samples_and_completes_without_truncation(tmp_path) -> None:
    config = _config(tmp_path)
    plan = sample_plan(sample_size=1)
    quality = next(item for item in plan.dimensions if item.dimension is Dimension.QUALITY)
    estimator = CostEstimator(PricingTable.load(Path(config.pricing.path)))
    first = estimator.estimate_grader(config.models.judge, spec=quality.graders[0], rows=ROWS)
    second = estimator.estimate_grader(config.models.judge, spec=quality.graders[1], rows=ROWS)
    budget = Budget(
        max_cost_usd=float(first + second / 2),
        max_wall_clock_s=60.0,
        max_target_requests=10,
    )
    runner = FakeRunner()
    deps = dependencies(tmp_path, planner=FakePlanner(plan=plan), runner=runner)
    manager = RunManager(config, dependencies=deps, enforce_budget=True)
    await manager.setup()
    worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir)
    run_id = await _start_and_approve(manager, worker, budget)

    assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET
    http, state = await _client(tmp_path, config)
    async with http:
        csrf = await login(http)
        extended = await http.post(
            f"/api/runs/{run_id}/budget/decision",
            json={
                "action": "extend",
                "budget": {
                    "max_cost_usd": float(first + second + Decimal("1")),
                    "max_wall_clock_s": 60.0,
                    "max_target_requests": 10,
                },
            },
            headers=csrf_headers(csrf),
        )
        assert extended.status_code == 202

    assert await worker.tick() is True
    assert (await manager.status(run_id)).status is RunStatus.COMPLETED
    report = await manager.report(run_id)
    assert report["truncated_dimensions"] == []
    assert report["dimensions"][0]["status"] == "ok"
    assert report["usage"]["cached"] == 1
    assert report["usage"]["fresh"] == 1
    assert runner.calls == 2
    await manager.close()
    await state.engine.dispose()
