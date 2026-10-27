# Eva-Agent

自主评测 agent：给定证据包、文本、数据集或在线 API，自动调查证据、制定评测计划、执行质量 / 耗时 / 成本 / 安全四维评测，产出带证据引用的报告与可复用 grader。

当前进度：**阶段 0（项目骨架与最小工作流）**。图只包含 `intake → plan → END`，使用确定性 `StubPlanner`，不调用模型。

## 开发环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync
cp config/config.example.yaml config/config.yaml
```

运行任何模型相关代码前，请在当前 shell 中导出所需密钥，例如 `export DEEPSEEK_API_KEY=...`。当前阶段尚未加载 `.env` 文件；`.env` 加载器会随后续阶段的应用入口一起加入。

## 检查与测试

```bash
uv run ruff check src tests
uv run ruff format --check src tests
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

设计规格见本地文件 `docs/superpowers/specs/2026-10-27-eva-agent-design.md`，阶段 0 计划见本地文件 `docs/superpowers/plans/2026-10-27-phase-0-scaffolding.md`。这些文档是本地工作笔记，不随仓库提交。
