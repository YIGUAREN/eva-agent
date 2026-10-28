# Eva-Agent

自主评测 agent：给定证据包、文本、数据集或在线 API，自动调查证据、制定评测计划、执行质量 / 耗时 / 成本 / 安全四维评测，产出带证据引用的报告与可复用 grader。

当前进度：**阶段 1（输入适配 + 证据工作区 + OpenJudge + quality 专家 + 报告 + 计划确认）**。图包含 `intake → plan → approve → dispatch → quality → merge → analyze → report`，运行前需要人工确认计划；upload 与 zip/tar 解压留待沙箱阶段。

## 开发环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync
cp config/config.example.yaml config/config.yaml
```

运行前请准备 `config/config.yaml` 与 `.env`；CLI 与演示脚本会通过 `python-dotenv` 加载 `.env`。

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
```

## 规格与计划

设计规格见本地文件 `docs/superpowers/specs/2026-10-27-eva-agent-design.md`，阶段 1 设计见 `docs/superpowers/specs/2026-10-27-phase-1-design.md`，阶段 1 计划见本地文件 `docs/superpowers/plans/2026-10-27-phase-1-dataset-e2e.md`。这些文档是本地工作笔记，不随仓库提交。
