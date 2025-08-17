from __future__ import annotations

from dataclasses import asdict
from typing import Dict
from pathlib import Path
import json
from datetime import datetime, timezone

from .models import Account, Position


def initialize_account(starting_cash: float) -> Account:
    return Account(cash=starting_cash, equity=starting_cash, realized_pnl=0.0, positions={})


def save_state(path: str, account: Account) -> None:
    serializable = {
        "cash": account.cash,
        "equity": account.equity,
        "realized_pnl": account.realized_pnl,
        "positions": {
            sym: {
                "qty": pos.qty,
                "avg_price": pos.avg_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "side": pos.side.value,
                "opened_ts": pos.opened_ts.isoformat() if pos.opened_ts else None,
                "updated_ts": pos.updated_ts.isoformat() if pos.updated_ts else None,
            }
            for sym, pos in account.positions.items()
        },
        "saved_ts": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(serializable, indent=2))


def load_state(path: str, default_account: Account) -> Account:
    p = Path(path)
    if not p.exists():
        return default_account
    data = json.loads(p.read_text())
    positions: Dict[str, Position] = {}
    for sym, d in data.get("positions", {}).items():
        pos = Position(symbol=sym)
        pos.qty = float(d.get("qty", 0.0))
        pos.avg_price = float(d.get("avg_price", 0.0))
        pos.unrealized_pnl = float(d.get("unrealized_pnl", 0.0))
        positions[sym] = pos
    return Account(
        cash=float(data.get("cash", default_account.cash)),
        equity=float(data.get("equity", default_account.equity)),
        realized_pnl=float(data.get("realized_pnl", default_account.realized_pnl)),
        positions=positions,
    )

def clear_state_file(path: str) -> None:
    p = Path(path)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


