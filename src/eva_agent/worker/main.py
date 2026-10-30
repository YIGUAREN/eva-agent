"""eva-worker entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from eva_agent.config.settings import load_config
from eva_agent.runtime.manager import RunManager
from eva_agent.worker.runner import RunWorker


def main() -> None:
    parser = argparse.ArgumentParser(prog="eva-worker")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    asyncio.run(_run(Path(args.config)))


async def _run(config_path: Path) -> None:
    config = load_config(config_path)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    except NotImplementedError:
        pass
    async with RunManager.open(config) as manager:
        worker = RunWorker.from_manager(
            manager,
            data_dir=config.storage.data_dir,
            poll_interval_s=config.web.poll_interval_s,
        )
        await worker.run_forever(stop)


if __name__ == "__main__":
    main()
