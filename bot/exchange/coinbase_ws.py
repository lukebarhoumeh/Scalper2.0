from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Callable

from websocket import WebSocketApp


@dataclass
class WSConfig:
    url: str = "wss://ws-feed.exchange.coinbase.com"
    heartbeat_sec: int = 30
    reconnect_max_backoff: int = 60


class CoinbaseWSClient:
    """Threaded WebSocket client with heartbeats, basic sequence tracking, and auto-reconnect.

    This client posts parsed ticker ticks into a callback provided by the caller.
    """

    def __init__(self, products: Iterable[str], on_tick: Callable[[str, float, float, float, float], None], config: Optional[WSConfig] = None) -> None:
        self.products = list(products)
        self.on_tick = on_tick
        self.cfg = config or WSConfig()
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_msg_ts = time.time()
        self._last_seq: Dict[str, int] = {}

    def _on_open(self, ws):
        sub_msg = {
            "type": "subscribe",
            "product_ids": self.products,
            "channels": ["ticker"],
        }
        ws.send(json.dumps(sub_msg))
        self._last_msg_ts = time.time()

    def _on_message(self, ws, message: str):
        self._last_msg_ts = time.time()
        try:
            data = json.loads(message)
            if data.get("type") != "ticker":
                return
            product_id = data.get("product_id")
            if not product_id:
                return
            bid = float(data.get("best_bid", 0))
            ask = float(data.get("best_ask", 0))
            price = float(data.get("price", 0)) if data.get("price") else (bid + ask) / 2.0 if bid and ask else 0.0
            seq = int(data.get("sequence", 0)) if data.get("sequence") is not None else 0
            server_time = data.get("time")
            # Simple gap detection: ensure sequence monotonic per product
            last = self._last_seq.get(product_id, 0)
            if seq and last and seq <= last:
                # Out of order; ignore
                return
            if seq:
                self._last_seq[product_id] = seq
            if price > 0 and bid > 0 and ask > 0:
                self.on_tick(product_id, price, bid, ask, self._last_msg_ts)
        except Exception:
            # swallow parsing errors; rely on heartbeat/reconnect
            pass

    def _on_error(self, ws, error):
        # Allow reconnect loop to handle
        pass

    def _on_close(self, ws, close_status_code, close_msg):
        # Allow reconnect loop to handle
        pass

    def start(self) -> None:
        def run_loop():
            backoff = 1.0
            while not self._stop.is_set():
                try:
                    self._ws = WebSocketApp(
                        self.cfg.url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                    )
                    self._ws.run_forever(ping_interval=self.cfg.heartbeat_sec, ping_timeout=10)
                except Exception:
                    pass
                # Reconnect with backoff
                if self._stop.wait(backoff):
                    break
                backoff = min(self.cfg.reconnect_max_backoff, backoff * 2)

        self._thread = threading.Thread(target=run_loop, name="coinbase_ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)


