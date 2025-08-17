from __future__ import annotations

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .models import Account, Position, Side


@dataclass
class RiskLimits:
    risk_per_trade: float
    max_total_exposure_usd: float
    max_daily_loss_percent: float
    max_symbol_usd: float = 1e12
    alt_exposure_usd_cap: float = 1e12


def compute_position_size_usd(account: Account, limits: RiskLimits) -> float:
    risk_dollars = max(0.0, account.equity) * limits.risk_per_trade
    return min(risk_dollars, limits.max_total_exposure_usd)


def circuit_breaker_triggered(account: Account, starting_equity: float, limits: RiskLimits) -> bool:
    if starting_equity <= 0:
        return False
    drop = (starting_equity - account.equity) / starting_equity
    return drop >= limits.max_daily_loss_percent


def current_total_exposure_usd(account: Account, last_prices: Dict[str, float]) -> float:
    exposure = 0.0
    for sym, pos in account.positions.items():
        price = last_prices.get(sym)
        if price is None:
            continue
        exposure += abs(pos.qty * price)
    return exposure


def current_symbol_exposure_usd(account: Account, last_prices: Dict[str, float], symbol: str) -> float:
    price = last_prices.get(symbol)
    if price is None:
        return 0.0
    pos = account.positions.get(symbol)
    qty = pos.qty if pos else 0.0
    return abs(qty * price)


def current_alt_exposure_usd(account: Account, last_prices: Dict[str, float]) -> float:
    """Sum exposure for non-BTC assets (treat anything not starting with BTC as 'alt')."""
    exposure = 0.0
    for sym, pos in account.positions.items():
        price = last_prices.get(sym)
        if price is None:
            continue
        base = sym.split("-")[0].upper()
        if base != "BTC":
            exposure += abs(pos.qty * price)
    return exposure


def can_increase_exposure(
    account: Account,
    last_prices: Dict[str, float],
    limits: RiskLimits,
    add_usd: float,
    symbol: Optional[str] = None,
) -> bool:
    """Check total, per-symbol, and sector (alt) caps before increasing exposure.

    Args:
        add_usd: proposed additional notional exposure in USD for this order
        symbol: optional symbol for per-symbol and alt cap checks
    """
    # Total exposure cap
    total_after = current_total_exposure_usd(account, last_prices) + max(add_usd, 0.0)
    if total_after > limits.max_total_exposure_usd:
        return False

    if symbol:
        price = last_prices.get(symbol)
        if price is not None:
            # Per-symbol cap
            pos = account.positions.get(symbol)
            current_symbol_usd = abs((pos.qty if pos else 0.0) * price)
            if current_symbol_usd + add_usd > limits.max_symbol_usd:
                return False
            # Alt-sector cap (non-BTC)
            base = symbol.split("-")[0].upper()
            if base != "BTC":
                alt_after = current_alt_exposure_usd(account, last_prices) + max(add_usd, 0.0)
                if alt_after > limits.alt_exposure_usd_cap:
                    return False
    return True


# ───────────────────────────────── Volatility targeting ─────────────────────────────────

def _stddev(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return var ** 0.5


def realized_volatility_pct(prices: List[float], window: int = 20) -> float:
    """Compute simple realized volatility as std of log returns over window (in percent).
    Returns a percentage (e.g., 0.8 means 0.8%).
    """
    if len(prices) < window + 1:
        return 0.0
    rets: List[float] = []
    start = len(prices) - window
    for i in range(start + 1, len(prices)):
        p0 = max(prices[i - 1], 1e-12)
        p1 = max(prices[i], 1e-12)
        rets.append((p1 / p0) - 1.0)
    sigma = _stddev(rets)
    return float(sigma * 100.0)


def compute_vol_targeted_qty(
    account: Account,
    limits: RiskLimits,
    last_price: float,
    price_history: List[float],
    vol_window: int = 20,
    min_vol_pct_floor: float = 0.10,
    max_vol_pct_cap: float = 5.00,
    precomputed_vol_pct: Optional[float] = None,
) -> Tuple[float, float]:
    """Volatility targeting: qty = risk_dollars / (last_price * vol_pct)

    Returns (qty, vol_pct_used)
    """
    risk_dollars = max(0.0, account.equity) * limits.risk_per_trade
    if last_price <= 0 or risk_dollars <= 0:
        return 0.0, 0.0

    vol_pct = precomputed_vol_pct if precomputed_vol_pct is not None else realized_volatility_pct(price_history, window=vol_window)
    # Clamp vol within [floor, cap] to avoid pathological sizing
    vol_eff = max(min_vol_pct_floor, min(vol_pct, max_vol_pct_cap))
    # Convert percent to fraction
    vol_frac = vol_eff / 100.0
    denom = last_price * max(vol_frac, 1e-6)
    qty = risk_dollars / denom
    return float(qty), float(vol_eff)


