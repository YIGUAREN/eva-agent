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

## Web UI（阶段 4）

```bash
# 1. 生成管理员密码哈希，并导出为环境变量（不要提交）
uv run python scripts/hash_password.py
export ADMIN_PASSWORD_HASH='$argon2id$...'

# 2. 构建前端
cd web && pnpm install && pnpm build && cd ..

# 3. 启动 API（自动执行 alembic upgrade head）
uv run eva-web --config config/config.yaml --port 8000

# 4. 另一个终端启动 worker（唯一执行 run 的进程）
uv run eva-worker --config config/config.yaml
```

公开页：`http://127.0.0.1:8000/share/<run_id>`（需先发布）。开发前端时用 `cd web && pnpm dev`，Vite 会把 `/api` 代理到 `127.0.0.1:8000`。

> `public_base_url` 必须与浏览器访问站点时使用的 Origin 完全一致（按本文档流程即 `http://127.0.0.1:8000`）。登录端点会校验 `Origin`/`Referer`，不一致会返回 403。Vite 开发服务器会把代理请求的 `Origin` 改写为 `http://127.0.0.1:8000`，因此 `pnpm dev` 下登录可以正常工作。

> 本地用 `http://127.0.0.1:8000` 做明文 HTTP 验收时，请先在 `config/config.yaml` 中把 `web.cookie_secure` 设为 `false`，否则浏览器不会保存登录 Cookie。生产环境必须保持 `true`（需要 TLS）。

### 上传证据包（阶段 4b）

1. 启动 sandbox-runner（需与 web/worker 共享 `data/` 卷）：

   ```bash
   export SANDBOX_TOKEN='<random 32+ chars>'
   uv run eva-sandbox --config config/config.yaml --port 8090
   ```

2. 管理员登录后调用：

   ```bash
   curl -X POST http://127.0.0.1:8000/api/uploads \
     -H "Cookie: eva_session=<session>" -H "X-CSRF-Token: <csrf>" \
     -F "file=@evidence.zip"
   ```

   返回 `upload_id`；`GET /api/uploads/{upload_id}` 查看解压状态与 manifest。zip/tar 均按魔数识别，任何路径穿越、链接、特殊文件或超限包整包拒绝，且不保留解压文件。

3. 建 run 时使用 `{"input": {"kind": "upload", "upload_id": "<id>"}}`。

证据包根目录可包含 `qa.jsonl`（每行一个带字符串 `query` 与 `response` 的 JSON 对象，可选 `context`/`reference`）供 quality 专家评分；没有根级 `qa.jsonl` 的包会跳过 quality 与 security 维度并在报告中标注为 skipped。
