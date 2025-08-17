from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class StrategyParams:
    rsi_buy: int
    rsi_sell: int
    sma_fast: int
    sma_slow: int
    max_spread_bps: int
    take_profit_bps: int
    stop_loss_bps: int
    trailing_stop_bps: int
    pos_size_multiplier: float = 1.0


@dataclass
class ParamBounds:
    rsi_buy: Tuple[int, int] = (20, 70)
    rsi_sell: Tuple[int, int] = (30, 80)
    sma_fast: Tuple[int, int] = (5, 20)
    sma_slow: Tuple[int, int] = (15, 60)
    max_spread_bps: Tuple[int, int] = (50, 300)
    take_profit_bps: Tuple[int, int] = (15, 120)
    stop_loss_bps: Tuple[int, int] = (20, 180)
    trailing_stop_bps: Tuple[int, int] = (15, 150)
    pos_size_multiplier: Tuple[float, float] = (0.5, 1.5)


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


class DynamicConfigManager:
    def __init__(
        self,
        initial: StrategyParams,
        history_path: str,
        bounds: Optional[ParamBounds] = None,
        cooldown_updates: int = 10,
        activity_target_tph: int = 60,
    ) -> None:
        self.params = initial
        self.bounds = bounds or ParamBounds()
        self.history_path = Path(history_path)
        self._updates_since_last = 0
        self._cooldown = max(3, cooldown_updates)
        self._activity_target_tph = max(1, activity_target_tph)

    def _persist(self, reason: str, meta: Dict) -> None:
        try:
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "params": asdict(self.params),
                "reason": reason,
                "meta": meta,
            }
            if not self.history_path.parent.exists():
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if self.history_path.exists():
                existing = json.loads(self.history_path.read_text() or "[]")
            existing.append(rec)
            self.history_path.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    def get_params(self) -> StrategyParams:
        return self.params

    def pos_multiplier(self) -> float:
        return float(_clip(self.params.pos_size_multiplier, *self.bounds.pos_size_multiplier))

    def maybe_update(
        self,
        session_win_rate_pct: float,
        session_sharpe: float,
        avg_markout_5s_bps: Optional[float],
        avg_spread_bps: Optional[float],
        avg_vol_pct: Optional[float],
        activity_tph: Optional[float] = None,
    ) -> bool:
        """Heuristic, bounded updates with cooldown and minimum sample.
        Adjust small deltas toward improving resiliency and expectancy.
        """
        self._updates_since_last += 1
        if self._updates_since_last < self._cooldown:
            return False
        self._updates_since_last = 0

        changed = False
        p = self.params
        meta: Dict[str, float] = {}

        # Spread gate tuning: tighten when spreads are tight and markouts positive; loosen when wide and negative
        if avg_spread_bps is not None:
            if avg_spread_bps < 30 and (avg_markout_5s_bps or 0) > 0:
                old = p.max_spread_bps
                p.max_spread_bps = int(_clip(p.max_spread_bps - 5, *self.bounds.max_spread_bps))
                changed |= p.max_spread_bps != old
            elif avg_spread_bps > 120 and (avg_markout_5s_bps or 0) < 0:
                old = p.max_spread_bps
                p.max_spread_bps = int(_clip(p.max_spread_bps + 5, *self.bounds.max_spread_bps))
                changed |= p.max_spread_bps != old

        # Activity targeting: if trade rate below target, loosen gates; if far above, tighten
        if activity_tph is not None:
            if activity_tph < 0.6 * self._activity_target_tph:
                # loosen slightly
                rb_old, rs_old = p.rsi_buy, p.rsi_sell
                p.rsi_buy = int(_clip(p.rsi_buy + 1, *self.bounds.rsi_buy))
                p.rsi_sell = int(_clip(p.rsi_sell - 1, *self.bounds.rsi_sell))
                ms_old = p.max_spread_bps
                p.max_spread_bps = int(_clip(p.max_spread_bps + 5, *self.bounds.max_spread_bps))
                changed |= (p.rsi_buy != rb_old or p.rsi_sell != rs_old or p.max_spread_bps != ms_old)
            elif activity_tph > 1.5 * self._activity_target_tph:
                # tighten slightly
                rb_old, rs_old = p.rsi_buy, p.rsi_sell
                p.rsi_buy = int(_clip(p.rsi_buy - 1, *self.bounds.rsi_buy))
                p.rsi_sell = int(_clip(p.rsi_sell + 1, *self.bounds.rsi_sell))
                ms_old = p.max_spread_bps
                p.max_spread_bps = int(_clip(p.max_spread_bps - 5, *self.bounds.max_spread_bps))
                changed |= (p.rsi_buy != rb_old or p.rsi_sell != rs_old or p.max_spread_bps != ms_old)

        # Volatility-aware exits: scale TP/SL/trailing with vol
        if avg_vol_pct is not None:
            if avg_vol_pct > 1.5:  # high vol
                tp_old, sl_old, tr_old = p.take_profit_bps, p.stop_loss_bps, p.trailing_stop_bps
                p.take_profit_bps = int(_clip(p.take_profit_bps + 5, *self.bounds.take_profit_bps))
                p.stop_loss_bps = int(_clip(p.stop_loss_bps + 10, *self.bounds.stop_loss_bps))
                p.trailing_stop_bps = int(_clip(p.trailing_stop_bps + 5, *self.bounds.trailing_stop_bps))
                changed |= (p.take_profit_bps != tp_old or p.stop_loss_bps != sl_old or p.trailing_stop_bps != tr_old)
            elif avg_vol_pct < 0.5:  # low vol
                tp_old, sl_old, tr_old = p.take_profit_bps, p.stop_loss_bps, p.trailing_stop_bps
                p.take_profit_bps = int(_clip(p.take_profit_bps - 5, *self.bounds.take_profit_bps))
                p.stop_loss_bps = int(_clip(p.stop_loss_bps - 5, *self.bounds.stop_loss_bps))
                p.trailing_stop_bps = int(_clip(p.trailing_stop_bps - 5, *self.bounds.trailing_stop_bps))
                changed |= (p.take_profit_bps != tp_old or p.stop_loss_bps != sl_old or p.trailing_stop_bps != tr_old)

        # RSI/SMA nudges based on win rate and Sharpe
        if session_win_rate_pct > 55 and session_sharpe > 0.5:
            # Slightly more selective entries
            rb_old, rs_old = p.rsi_buy, p.rsi_sell
            p.rsi_buy = int(_clip(p.rsi_buy - 1, *self.bounds.rsi_buy))
            p.rsi_sell = int(_clip(p.rsi_sell + 1, *self.bounds.rsi_sell))
            changed |= (p.rsi_buy != rb_old or p.rsi_sell != rs_old)
        elif session_win_rate_pct < 45 and session_sharpe < 0.0:
            # Loosen thresholds a touch to find more edges
            rb_old, rs_old = p.rsi_buy, p.rsi_sell
            p.rsi_buy = int(_clip(p.rsi_buy + 1, *self.bounds.rsi_buy))
            p.rsi_sell = int(_clip(p.rsi_sell - 1, *self.bounds.rsi_sell))
            changed |= (p.rsi_buy != rb_old or p.rsi_sell != rs_old)

        # Position size multiplier guardrails using markout
        if avg_markout_5s_bps is not None:
            old = p.pos_size_multiplier
            if avg_markout_5s_bps > 2.0:
                p.pos_size_multiplier = _clip(p.pos_size_multiplier * 1.05, *self.bounds.pos_size_multiplier)
            elif avg_markout_5s_bps < -2.0:
                p.pos_size_multiplier = _clip(p.pos_size_multiplier * 0.95, *self.bounds.pos_size_multiplier)
            changed |= p.pos_size_multiplier != old

        if changed:
            meta.update(
                dict(
                    win_rate=session_win_rate_pct,
                    sharpe=session_sharpe,
                    avg_markout_5s_bps=avg_markout_5s_bps or 0.0,
                    avg_spread_bps=avg_spread_bps or 0.0,
                    avg_vol_pct=avg_vol_pct or 0.0,
                )
            )
            self._persist("update", meta)
        return changed


