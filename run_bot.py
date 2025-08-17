#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import signal

from bot.config import load_config
from bot.app import App


async def main() -> None:
    cfg = load_config()
    app = App(cfg)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.request_shutdown)

    await app.run()


if __name__ == "__main__":
    asyncio.run(main())


