"""Import every run bundle in a directory; used to seed showcase data."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eva_agent.config.settings import load_config
from eva_agent.packages.run_bundle import ImportResult, import_run_bundle


async def seed(config, directory: Path, *, public: bool) -> list[ImportResult]:
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}")
    results: list[ImportResult] = []
    for archive in sorted(directory.glob("*.zip")):
        results.append(await import_run_bundle(config=config, archive=archive, public=public))
    return results


async def _main(directory: Path, config_path: Path, public: bool) -> int:
    config = load_config(config_path)
    results = await seed(config, directory, public=public)
    print(json.dumps([item.__dict__ for item in results], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="demo_runs")
    parser.add_argument("--config", default="config/config.yaml")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--public", dest="public", action="store_true", default=True)
    visibility.add_argument("--private", dest="public", action="store_false")
    args = parser.parse_args()
    return asyncio.run(_main(Path(args.dir), Path(args.config), args.public))


if __name__ == "__main__":
    raise SystemExit(main())
