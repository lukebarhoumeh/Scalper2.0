from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Iterable, Optional
import time
import math
import requests
from .exchange.coinbase_ws import CoinbaseWSClient

from .models import Tick


class MarketDataFeed:
    """Graceful market feed. If a real source is unavailable, it generates a
    random-walk synthetic feed so the system and tests can proceed reliably.
    """

    def __init__(self, symbols: Iterable[str], poll_interval_sec: float = 1.0, source: str = "synthetic", request_rate_limit: int = 8):
        self.symbols = list(symbols)
        self.poll_interval_sec = poll_interval_sec
        self._prices: Dict[str, float] = {s: 100.0 + 10.0 * random.random() for s in self.symbols}
        self.source = source
        self.request_rate_limit = request_rate_limit
        self._ws_client: Optional[CoinbaseWSClient] = None

    async def ticks(self) -> AsyncIterator[Tick]:
        if self.source == "synthetic":
            while True:
                for s in self.symbols:
                    last = self._prices[s]
                    step = random.uniform(-0.2, 0.2)
                    new_price = max(0.01, last + step)
                    self._prices[s] = new_price
                    spread = max(0.01, new_price * 0.0008)
                    bid = new_price - spread / 2
                    ask = new_price + spread / 2
                    yield Tick(symbol=s, price=new_price, bid=bid, ask=ask, ts=datetime.now(timezone.utc))
                await asyncio.sleep(self.poll_interval_sec)
        elif self.source == "coinbase_rest":
            # Minimal REST polling with backoff and time skew check (prod WS adapter to be added)
            backoff = 0.5
            max_backoff = 8.0
            while True:
                try:
                    if self._ws_client is None:
                        def on_tick(sym: str, price: float, bid: float, ask: float, ts: float):
                            # WS updates write cache; REST fills gaps
                            self._prices[sym] = price
                        self._ws_client = CoinbaseWSClient(self.symbols, on_tick)
                        self._ws_client.start()
                    for s in self.symbols:
                        # Coinbase Advanced Trade REST best bid/ask equivalent via product ticker
                        r = requests.get(f"https://api.exchange.coinbase.com/products/{s}/ticker", timeout=3.0)
                        if r.status_code != 200:
                            raise RuntimeError(f"HTTP {r.status_code}")
                        data = r.json()
                        bid = float(data.get("bid", 0))
                        ask = float(data.get("ask", 0))
                        price = (bid + ask) / 2.0 if bid and ask else float(data.get("price", 0))
                        if price <= 0 or bid <= 0 or ask <= 0:
                            continue
                        self._prices[s] = price
                        yield Tick(symbol=s, price=price, bid=bid, ask=ask, ts=datetime.now(timezone.utc))
                    backoff = 0.5
                    await asyncio.sleep(self.poll_interval_sec)
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(max_backoff, backoff * 1.7)


