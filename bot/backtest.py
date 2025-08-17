from __future__ import annotations

import asyncio
from typing import Iterable, List
from datetime import datetime, timezone

from .models import Tick
from .app import App
from .config import load_config
import csv


class HistoricalFeed:
    def __init__(self, rows: List[dict]):
        self.rows = rows

    async def ticks(self):  # Async iterator compatible with App
        for r in self.rows:
            yield Tick(
                symbol=r["symbol"],
                price=float(r["price"]),
                bid=float(r.get("bid", r["price"])) ,
                ask=float(r.get("ask", r["price"])) ,
                ts=datetime.fromisoformat(r["ts"]) if isinstance(r["ts"], str) else r["ts"],
            )


async def run_backtest(rows: List[dict]) -> None:
    cfg = load_config()
    app = App(cfg)
    # Monkey patch market feed with historical one
    app.market.ticks = HistoricalFeed(rows).ticks  # type: ignore
    await app.run()


def load_csv_rows(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "symbol": row["symbol"],
                "price": float(row["price"]),
                "bid": float(row.get("bid", row["price"])),
                "ask": float(row.get("ask", row["price"])),
                "ts": row.get("ts", datetime.now(timezone.utc).isoformat()),
            })
    return out



