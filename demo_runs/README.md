# Demo runs

本目录用于放置 `*.run-bundle.zip` 运行包，供 `scripts/seed_demo_runs.py` 导入为演示数据。

## 生成运行包

运行包由 `eva export-run` 生成，且只适用于已完成并已公开的 run：

```bash
uv run eva export-run <run_id> --out demo_runs/
```

生成真实的运行包需要配置好的模型凭据，以及一次已完成并发布的 run。仓库默认只保留本文件与 `.gitkeep`，不附带任何真实运行包；是否提交运行包是可选的，但绝不能包含密钥、令牌或任何绝对路径（本地路径、内部地址、数据目录路径等）。

## 导入

```bash
uv run python scripts/seed_demo_runs.py --dir demo_runs --public
```

脚本按文件名顺序导入目录中的每个运行包，并统一通过 `import_run_bundle` 完成；该接口按 `run_id` 幂等，同名 run 已存在时报告 `skipped`，不会覆盖。`--public` 决定导入的 run 是否公开（默认公开），`--private` 则保持私有。

## 安全说明

运行包是**管理员级产物**：它有意包含 `usage.spent_usd` 与本次 run 的预算上限（budget caps），这些字段不会通过公开 API 暴露。请只在受控环境中导出、存放和导入运行包，不要把内部运行包当作可公开分发的文件。
