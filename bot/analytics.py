from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
from math import sqrt

from .models import TradeRecord


@dataclass
class KPIs:
    win_rate_pct: float
    avg_r: float
    sharpe: float
    max_drawdown_pct: float
    fee_pct: float
    turnover_usd: float


def compute_kpis(trades: List[TradeRecord]) -> KPIs:
    if not trades:
        return KPIs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Simple approximations using per-trade deltas
    pnls: List[float] = []
    fees: float = 0.0
    turnover: float = 0.0
    wins = 0
    for tr in trades:
        # realized PnL at the time of this trade relative to previous trade
        pnls.append(tr.realized_pnl)
        turnover += tr.fill.qty * tr.fill.price
        # Fee tracking is applied in broker PnL already; approximate fee% via notional * 0.0001 per trade if needed
    # Win rate using monotonic increases in realized PnL between trades
    realized_changes = [pnls[i] - pnls[i - 1] for i in range(1, len(pnls))]
    if realized_changes:
        wins = sum(1 for d in realized_changes if d > 0)
        losses = len(realized_changes) - wins
        win_rate = (wins / max(1, len(realized_changes))) * 100.0
        avg_return = sum(realized_changes) / max(1, len(realized_changes))
        # Sharpe (naive): mean/std of per-trade PnL (assumes unit variance of time)
        mean = avg_return
        var = sum((d - mean) ** 2 for d in realized_changes) / max(1, len(realized_changes))
        std = sqrt(var) if var > 0 else 0.0
        sharpe = (mean / std) if std > 0 else 0.0
        # Max drawdown on cumulative PnL series
        peak = -1e18
        max_dd = 0.0
        cum = 0.0
        for d in realized_changes:
            cum += d
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)
        max_dd_pct = abs(max_dd) / max(1.0, abs(peak)) * 100.0 if peak != 0 else 0.0
    else:
        win_rate = 0.0
        sharpe = 0.0
        avg_return = 0.0
        max_dd_pct = 0.0

    return KPIs(win_rate_pct=win_rate, avg_r=avg_return, sharpe=sharpe, max_drawdown_pct=max_dd_pct, fee_pct=0.0, turnover_usd=turnover)


