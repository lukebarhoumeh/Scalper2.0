from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import csv
from itertools import product

from .backtest import load_csv_rows, run_backtest, HistoricalFeed
from .config import load_config
from .app import App


@dataclass
class SweepSpec:
    rsi_buy_values: List[int]
    rsi_sell_values: List[int]
    sma_fast_values: List[int]
    sma_slow_values: List[int]
    max_spread_bps_values: List[int]


async def run_one(rows: List[dict], cfg_overrides: dict) -> Tuple[float, dict]:
    cfg = load_config()
    # Override selected fields
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    app = App(cfg)
    app.market.ticks = HistoricalFeed(rows).ticks  # type: ignore
    await app.run()
    return app.account.realized_pnl, cfg_overrides


async def sweep(csv_path: str, spec: SweepSpec, out_csv: str) -> None:
    rows = load_csv_rows(csv_path)
    combos = list(product(spec.rsi_buy_values, spec.rsi_sell_values, spec.sma_fast_values, spec.sma_slow_values, spec.max_spread_bps_values))
    results: List[Tuple[float, dict]] = []
    for rb, rs, sf, ss, ms in combos:
        pnl, cfg = await run_one(rows, {
            'rsi_buy': rb, 'rsi_sell': rs, 'sma_fast': sf, 'sma_slow': ss, 'spread_bps_max': ms,
        })
        results.append((pnl, cfg))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pnl','rsi_buy','rsi_sell','sma_fast','sma_slow','max_spread_bps'])
        w.writeheader()
        for pnl, cfg in sorted(results, key=lambda x: x[0], reverse=True):
            w.writerow({'pnl': pnl, 'rsi_buy': cfg['rsi_buy'], 'rsi_sell': cfg['rsi_sell'], 'sma_fast': cfg['sma_fast'], 'sma_slow': cfg['sma_slow'], 'max_spread_bps': cfg['spread_bps_max']})


