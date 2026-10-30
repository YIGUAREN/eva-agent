# Eva-Agent

自主评测 agent：给定证据包、文本、数据集或在线 API，自动调查证据、制定评测计划、执行质量 / 耗时 / 成本 / 安全四维评测，产出带证据引用的报告与可复用 grader。

当前进度：**阶段 3（security 专家 + 授权策略 + 审计）**。图包含 `intake → plan → approve → probe → dispatch → {quality | latency | cost | security} → merge → analyze → report`；target 输入在批准后共享一次 probe，四维并行执行。security 先跑 OpenJudge 被动检查（`harmfulness`，检测到代码特征时加 `code_security`），仅在 demo/allowlist、管理员授权、预算与超时都满足时才调用 AIG 主动扫描。upload 与 zip/tar 解压留待沙箱阶段。

## 开发环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync
cp config/config.example.yaml config/config.yaml
```

运行前请准备 `config/config.yaml` 与 `.env`；CLI 与演示脚本会通过 `python-dotenv` 加载 `.env`。

## Demo target

```bash
uv run uvicorn demo_target.app:app --host 127.0.0.1 --port 8100
uv run python scripts/phase2_demo.py \
  --target demo \
  --prompt "如何重置密码？" \
  --prompt "退款多久到账？" \
  --max-cost-usd 0.5 \
  --max-wall-clock-s 600 \
  --max-target-requests 5 \
  --yes
```

## AIG 开发栈

本地 AIG 与 demo target 通过 Compose 启动：

```bash
docker compose -f deploy/compose.dev.yaml up -d
```

AIG 只绑定 `127.0.0.1:8088`；镜像 tag 固定在 `deploy/compose.dev.yaml`。主动扫描的授权由 `config/security_policy.yaml` 控制，默认只对 `demo` 目标开放：

```yaml
active_scan_enabled: true
requester_is_admin: true
authorization_ref: "demo"
demo_targets:
  - name: demo
    origin: "http://127.0.0.1:8100"
    aig_base_url: "http://demo-target:8100/v1"
allowlist: []
```

真实 AIG 冒烟测试默认排除，显式运行：

```bash
uv run pytest -m aig -q
```

## 检查与测试

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pyright
uv run pytest --cov=eva_agent --cov-report=term-missing --cov-fail-under=80
```

## 目录

```text
src/eva_agent/
├── adapters/    # 外部输入与引擎适配
├── config/      # YAML 配置加载
├── domain/      # 领域对象（Budget / RunInput / Evidence / EvalPlan）
├── graph/       # LangGraph 状态、节点、构建
├── llm/         # 模型工厂
├── storage/     # SQLAlchemy 表与仓库
└── ...
tests/
├── unit/
└── integration/
demo_target/     # 本地 OpenAI 兼容演示目标
```

## 规格与计划

设计规格见本地文件 `docs/superpowers/specs/2026-10-27-eva-agent-design.md`，阶段 1 设计见 `docs/superpowers/specs/2026-10-27-phase-1-design.md`，阶段 2 设计见 `docs/superpowers/specs/2026-10-28-phase-2-target-eval-design.md`，阶段 1 计划见 `docs/superpowers/plans/2026-10-27-phase-1-dataset-e2e.md`，阶段 2 计划见 `docs/superpowers/plans/2026-10-28-phase-2-target-eval.md`。
