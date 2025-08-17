# ─────────────────── market_data.py ──────────────────────────────────────────
from __future__ import annotations

import json, logging, threading, time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import requests, websocket
from coinbase.rest import RESTClient
from dateutil import parser
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from config import (
    COINBASE_API_KEY,
    COINBASE_API_SECRET,
    REST_RATE_LIMIT_PER_S,
    TRADE_COINS,
    USE_WS_FEED,
)

_LOG = logging.getLogger(__name__)
# ───────────────────────── REST client & rate‑limiter ────────────────────────
_CLIENT: RESTClient | None = None
_LOCK = threading.Lock()
_LAST_CALLS = []                       # timestamps of last N REST hits

def _rate_limited() -> None:
    """Simple token‑bucket: max REST_RATE_LIMIT_PER_S within a rolling second."""
    now = time.time()
    _LAST_CALLS[:] = [t for t in _LAST_CALLS if now - t < 1.0]
    if len(_LAST_CALLS) >= REST_RATE_LIMIT_PER_S:
        time.sleep(1.0 - (now - _LAST_CALLS[0]))
    _LAST_CALLS.append(time.time())

def _client() -> RESTClient:
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            _CLIENT = RESTClient(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET)
        return _CLIENT

# ───────────────────────── WebSocket best‑bid/ask cache ──────────────────────
_BBO: Dict[str, Tuple[float, float]] = defaultdict(lambda: (np.nan, np.nan))

def _ws_loop(products: list[str]) -> None:
    url = "wss://advanced-trade-ws.coinbase.com"
    sub_msg = {
        "type": "subscribe",
        "product_ids": [f"{p}-USD" for p in products],
        "channel": "ticker",
    }

    def on_msg(_, msg: str):
        j = json.loads(msg)
        if j.get("channel") != "ticker" or "product_id" not in j:
            return
        bid = float(j["best_bid"])
        ask = float(j["best_ask"])
        _BBO[j["product_id"]] = (bid, ask)

    while True:            # auto‑reconnect
        try:
            ws = websocket.WebSocketApp(url, on_message=on_msg)
            ws.on_open = lambda ws: ws.send(json.dumps(sub_msg))
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            _LOG.warning("WS reconnect after error: %s", e, exc_info=False)
            time.sleep(5)

if USE_WS_FEED:
    threading.Thread(target=_ws_loop, args=(TRADE_COINS,), daemon=True).start()

# ───────────────────────── helpers & public API ──────────────────────────────
_RETRYABLE = (requests.exceptions.RequestException, ConnectionError)

def _retry():
    return retry(
        wait=wait_exponential(multiplier=0.4, min=0.8, max=8),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )

def mid_price(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0

@_retry()
def _rest_best_bid_ask(product_id: str) -> Tuple[float, float]:
    _rate_limited()
    pb = _client().get_best_bid_ask(product_ids=[product_id]).pricebooks[0]
    return float(pb.bids[0].price), float(pb.asks[0].price)

def get_best_bid_ask(product_id: str) -> Tuple[float, float]:
    bid, ask = _BBO.get(product_id, (np.nan, np.nan))
    if np.isnan(bid) or np.isnan(ask):
        bid, ask = _rest_best_bid_ask(product_id)  # cold‑start / WS gap‑filler
        _BBO[product_id] = (bid, ask)
    return bid, ask

def get_last_price(product_id: str) -> float:
    bid, ask = get_best_bid_ask(product_id)
    return mid_price(bid, ask)

@_retry()
def get_historic_candles(product_id: str, granularity_sec=60, lookback=300) -> pd.DataFrame:
    _rate_limited()
    gran_map = {
        60: "ONE_MINUTE", 300: "FIVE_MINUTE", 900: "FIFTEEN_MINUTE",
        1800: "THIRTY_MINUTE", 3600: "ONE_HOUR", 7200: "TWO_HOUR",
        21600: "SIX_HOUR", 86400: "ONE_DAY",
    }
    gran = gran_map[granularity_sec]
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity_sec * lookback)
    resp = _client().get_candles(
        product_id=product_id,
        start=str(int(start.timestamp())),
        end=str(int(end.timestamp())),
        granularity=gran,
    )
    if not resp.candles:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.DataFrame({
        "time":   [datetime.fromtimestamp(int(c.start), tz=timezone.utc) for c in resp.candles],
        "open":   [float(c.open)   for c in resp.candles],
        "high":   [float(c.high)   for c in resp.candles],
        "low":    [float(c.low)    for c in resp.candles],
        "close":  [float(c.close)  for c in resp.candles],
        "volume": [float(c.volume) for c in resp.candles],
    }).sort_values("time").reset_index(drop=True)
    return df

def realised_volatility(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    log_ret = np.log1p(df["close"].pct_change().dropna())
    ann = 365*24*60*60 / (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
    return float(np.sqrt(log_ret.var() * ann) * 100)
