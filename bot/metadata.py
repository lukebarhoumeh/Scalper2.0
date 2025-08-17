from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SymbolMetadata:
    symbol: str
    fee_bps: float           # taker fee in basis points (e.g., 10 = 0.10%)
    min_notional_usd: float  # minimum order notional
    tick_size: float         # price increment
    lot_size: float          # quantity increment


DEFAULT_METADATA: Dict[str, SymbolMetadata] = {
    "BTC-USD": SymbolMetadata(symbol="BTC-USD", fee_bps=10.0, min_notional_usd=1.0, tick_size=0.01, lot_size=0.000001),
    "ETH-USD": SymbolMetadata(symbol="ETH-USD", fee_bps=10.0, min_notional_usd=1.0, tick_size=0.01, lot_size=0.00001),
    "SOL-USD": SymbolMetadata(symbol="SOL-USD", fee_bps=10.0, min_notional_usd=1.0, tick_size=0.001, lot_size=0.001),
}


def get_symbol_metadata(symbol: str) -> SymbolMetadata:
    return DEFAULT_METADATA.get(symbol, SymbolMetadata(symbol=symbol, fee_bps=10.0, min_notional_usd=1.0, tick_size=0.01, lot_size=0.000001))


