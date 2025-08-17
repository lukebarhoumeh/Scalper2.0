from __future__ import annotations

import asyncio
import time
import random
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, Optional


@dataclass
class TokenBucket:
    rate_per_sec: int
    capacity: int
    tokens: float
    last: float

    def __init__(self, rate_per_sec: int, capacity: Optional[int] = None) -> None:
        self.rate_per_sec = max(1, rate_per_sec)
        self.capacity = capacity if capacity is not None else self.rate_per_sec
        self.tokens = float(self.capacity)
        self.last = time.time()

    async def acquire(self, tokens: int = 1) -> None:
        while True:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            await asyncio.sleep(max(0.0, (tokens - self.tokens) / self.rate_per_sec))


def backoff_gen(base: float = 0.5, factor: float = 2.0, jitter: float = 0.2, maximum: float = 30.0):
    delay = base
    while True:
        yield min(max(0.0, delay + random.uniform(-jitter, jitter) * delay), maximum)
        delay = min(delay * factor, maximum)


class MarketDataAdapter:
    """Scaffold for Coinbase WS primary + REST fallback.
    Not wired yet; placeholder to be integrated in a later milestone.
    """

    def __init__(self, pairs: Iterable[str], rate_limit: int = 8) -> None:
        self.pairs = list(pairs)
        self._tb = TokenBucket(rate_limit)
        self._connected = False
        self._last_seq: Dict[str, int] = {}
        self._last_ts: Dict[str, float] = {}

    async def connect(self) -> None:
        # TODO: implement WS auth/connect and subscription
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def best_bid_ask(self, symbol: str) -> Optional[tuple[float, float]]:
        # TODO: REST fallback with rate limiting and backoff
        await self._tb.acquire(1)
        return None


class TradeAdapter:
    """Scaffold for Coinbase trading via REST; confirmations via WS.
    Not wired yet; placeholder to be integrated in a later milestone.
    """

    def __init__(self, api_key: Optional[str], api_secret: Optional[str], rate_limit: int = 8) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self._tb = TokenBucket(rate_limit)

    async def place_order(self, symbol: str, side: str, qty: float, order_type: str = "market") -> Dict:
        await self._tb.acquire(1)
        # TODO: implement REST call with retry/backoff
        return {"status": "submitted"}


