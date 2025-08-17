from __future__ import annotations

import time
from typing import Dict, List, Optional

from .models import Account, TradeRecord


ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_CYAN = "\033[96m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"


class TerminalUI:
    """Minimal, safe terminal UI that never assumes missing fields.
    Shows balances, positions, and recent activity, with simple colors.
    """

    def __init__(self, refresh_hz: float = 4.0):
        self.last_render_time = 0.0
        self.refresh_period = 1.0 / max(0.5, float(refresh_hz))

    def render(self, account: Account, recent: List[TradeRecord], stats: Optional[Dict[str, float]] = None, indicators: Optional[Dict[str, Dict[str, float | str]]] = None) -> None:
        now = time.time()
        if now - self.last_render_time < self.refresh_period:
            return
        self.last_render_time = now

        header = f"{ANSI_BOLD}{ANSI_CYAN}SCALPERBOT 2.0{ANSI_RESET}"
        print(f"\n=== {header} ===")
        eq_color = ANSI_GREEN if account.equity >= account.cash else ANSI_YELLOW
        print(
            f"Cash: ${account.cash:,.2f}  Equity: {eq_color}${account.equity:,.2f}{ANSI_RESET}  Realized PnL: "
            f"{(ANSI_GREEN if account.realized_pnl >= 0 else ANSI_RED)}${account.realized_pnl:,.2f}{ANSI_RESET}"
        )

        if stats:
            wr = stats.get("win_rate", 0.0)
            ticks = int(stats.get("ticks", 0))
            trades = int(stats.get("trades", 0))
            exposure = stats.get("exposure", 0.0)
            last_tick_age = stats.get("last_tick_age", None)
            conn_status = stats.get("connection", "synthetic")
            util_total = stats.get("util_total", None)
            util_alt = stats.get("util_alt", None)
            status_txt = f"{ANSI_BOLD}System:{ANSI_RESET} ticks={ticks} trades={trades} win_rate={wr:.1f}% exposure=${exposure:,.2f}"
            if last_tick_age is not None:
                status_txt += f" last_tick_age={last_tick_age:.1f}s"
            status_txt += f" source={conn_status}"
            if util_total is not None:
                status_txt += f" util_total={util_total*100:.1f}%"
            if util_alt is not None:
                status_txt += f" util_alt={util_alt*100:.1f}%"
            print(status_txt)

        print(f"{ANSI_BOLD}Positions:{ANSI_RESET}")
        if not account.positions:
            print("  (none)")
        for sym, pos in account.positions.items():
            ucolor = ANSI_GREEN if pos.unrealized_pnl >= 0 else ANSI_RED
            print(
                f"  {sym}: qty={pos.qty:.6f} avg={pos.avg_price:.4f} uPnL={ucolor}${pos.unrealized_pnl:.2f}{ANSI_RESET}"
            )
            if indicators and sym in indicators:
                m = indicators[sym]
                rs = m.get("rsi")
                sf = m.get("sma_fast")
                ss = m.get("sma_slow")
                sp = m.get("spread_bps")
                rs_txt = f"RSI={rs:.1f}" if isinstance(rs, (int, float)) else "RSI=?"
                sf_txt = f"SMA{int(sf) if isinstance(sf, (int, float)) else '?'}"
                ss_txt = f"SMA{int(ss) if isinstance(ss, (int, float)) else '?'}"
                sp_txt = f"spread={sp:.1f}bps" if isinstance(sp, (int, float)) else "spread=?"
                reason = m.get("reason") or ""
                last_side = m.get("last_side") or "?"
                print(f"      {rs_txt} {sf_txt}/{ss_txt} {sp_txt} last={last_side} reason={reason}")

        print(f"{ANSI_BOLD}Recent Activity:{ANSI_RESET}")
        for tr in recent[-5:]:
            side_str = tr.fill.side.value
            scolor = ANSI_GREEN if side_str == "buy" else ANSI_RED
            print(
                f"  {tr.fill.ts.isoformat()} {tr.fill.symbol} {scolor}{side_str.upper()}{ANSI_RESET} "
                f"{tr.fill.qty:.6f} @ {tr.fill.price:.4f} | realized=${account.realized_pnl:,.2f}"
            )


