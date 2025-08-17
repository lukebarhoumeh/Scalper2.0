from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
import csv
from pathlib import Path

from .models import Order, Fill, Side
from .metadata import get_symbol_metadata


@dataclass
class LedgerEntry:
    ts: datetime
    symbol: str
    side: Side
    qty: float
    price: float
    notional: float
    fee_usd: float
    slippage_bps: float
    markout_5s_bps: Optional[float] = None
    markout_30s_bps: Optional[float] = None
    markout_300s_bps: Optional[float] = None
    realized_pnl_after: Optional[float] = None


class Ledger:
    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []
        self._flushed: int = 0

    def add_fill(self, order: Order, fill: Fill, bid: float, ask: float, realized_after: float) -> LedgerEntry:
        meta = get_symbol_metadata(fill.symbol)
        fee = (meta.fee_bps / 1e4) * fill.price * fill.qty
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else fill.price
        # Assume market order executes near mid; use deviation as slippage
        slippage_bps = ((fill.price / mid) - 1.0) * 1e4 * (1.0 if fill.side == Side.BUY else -1.0)
        le = LedgerEntry(
            ts=fill.ts,
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty,
            price=fill.price,
            notional=fill.qty * fill.price,
            fee_usd=fee,
            slippage_bps=slippage_bps,
            realized_pnl_after=realized_after,
        )
        self.entries.append(le)
        return le

    def update_markouts(self, last_prices: Dict[str, float], now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        for e in self.entries:
            px_now = last_prices.get(e.symbol)
            if not px_now or e.price <= 0:
                continue
            age = (now - e.ts).total_seconds()
            direction = 1.0 if e.side == Side.BUY else -1.0
            bps = ((px_now / e.price) - 1.0) * 1e4 * direction
            if e.markout_5s_bps is None and age >= 5:
                e.markout_5s_bps = bps
            if e.markout_30s_bps is None and age >= 30:
                e.markout_30s_bps = bps
            if e.markout_300s_bps is None and age >= 300:
                e.markout_300s_bps = bps

    def to_rows(self) -> List[Dict]:
        return [asdict(e) for e in self.entries]

    def write_csv(self, path: str) -> None:
        rows = self.to_rows()
        if not rows:
            return
        p = Path(path)
        is_new = not p.exists()
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            if is_new:
                w.writeheader()
            for r in rows:
                r = {**r, "ts": r["ts"].isoformat() if isinstance(r["ts"], datetime) else r["ts"]}
                w.writerow(r)

    def flush_to_csv(self, path: str) -> None:
        """Write only new entries since last flush to CSV and advance the flush pointer."""
        if self._flushed >= len(self.entries):
            return
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        new_entries = self.entries[self._flushed :]
        if not new_entries:
            return
        is_new = not p.exists()
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=asdict(new_entries[0]).keys())
            if is_new:
                w.writeheader()
            for e in new_entries:
                r = asdict(e)
                r["ts"] = r["ts"].isoformat() if isinstance(r["ts"], datetime) else r["ts"]
                w.writerow(r)
        self._flushed = len(self.entries)


