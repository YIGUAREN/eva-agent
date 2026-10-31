"""eva-sandbox entrypoint: extract untrusted evidence packages only."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from eva_agent.config.settings import ConfigError, load_config, require_sandbox_token
from eva_agent.sandbox.app import create_sandbox_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="eva-sandbox")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    try:
        token = require_sandbox_token()
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None
    app = create_sandbox_app(data_dir=config.storage.data_dir, token=token, uploads=config.uploads)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", proxy_headers=False)


if __name__ == "__main__":
    main()
