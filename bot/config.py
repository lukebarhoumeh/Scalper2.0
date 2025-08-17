from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os

try:
    # Optional: load .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class BotConfig:
    trading_pairs: List[str]
    paper_trading: bool
    base_position_usd: float
    max_position_usd: float
    max_total_exposure_usd: float
    starting_cash_usd: float
    rsi_buy: int
    rsi_sell: int
    sma_fast: int
    sma_slow: int
    spread_bps_max: int
    risk_per_trade: float
    max_daily_loss_percent: float
    circuit_breaker_errors: int
    poll_interval_sec: float
    ui_refresh_hz: float
    state_path: str
    logs_dir: str
    take_profit_bps: int
    stop_loss_bps: int
    trailing_stop_bps: int
    data_source: str  # synthetic | coinbase_rest
    coinbase_api_key: str | None
    coinbase_api_secret: str | None
    max_requests_per_sec: int
    alt_exposure_usd_cap: float
    reset_state_on_start: bool
    tuning_enabled: bool
    activity_target_tph: int
    ledger_flush_sec: int
    enable_mean_reversion: bool
    enable_breakout: bool
    enable_grid_overlay: bool


def _get_env_list(name: str, default_csv: str) -> List[str]:
    value = os.getenv(name, default_csv)
    return [s.strip() for s in value.split(",") if s.strip()]


def load_config() -> BotConfig:
    return BotConfig(
        trading_pairs=_get_env_list(
            "TRADING_PAIRS", os.getenv("TRADING_PAIRS", "BTC-USD,ETH-USD,SOL-USD")
        ),
        paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
        base_position_usd=float(os.getenv("BASE_POSITION_SIZE", "50")),
        max_position_usd=float(os.getenv("MAX_POSITION_SIZE", "200")),
        max_total_exposure_usd=float(os.getenv("INVENTORY_CAP_USD", "2000")),
        starting_cash_usd=float(os.getenv("STARTING_CAPITAL", "10000")),
        rsi_buy=int(os.getenv("RSI_BUY_THRESHOLD", "35")),
        rsi_sell=int(os.getenv("RSI_SELL_THRESHOLD", "65")),
        sma_fast=int(os.getenv("SMA_FAST", "8")),
        sma_slow=int(os.getenv("SMA_SLOW", "21")),
        spread_bps_max=int(os.getenv("MAX_SPREAD_BPS", "150")),
        risk_per_trade=float(os.getenv("RISK_PERCENT", "0.01")),
        max_daily_loss_percent=float(os.getenv("MAX_DAILY_LOSS_PERCENT", "0.1")),
        circuit_breaker_errors=int(os.getenv("MAX_ERRORS", "20")),
        poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "1.0")),
        ui_refresh_hz=float(os.getenv("UI_REFRESH_HZ", "4")),
        state_path=os.getenv("STATE_PATH", "bot_state.json"),
        logs_dir=os.getenv("LOGS_DIR", "logs"),
        take_profit_bps=int(os.getenv("TAKE_PROFIT_BPS", "25")),
        stop_loss_bps=int(os.getenv("STOP_LOSS_BPS", "35")),
        trailing_stop_bps=int(os.getenv("TRAILING_STOP_BPS", "30")),
        data_source=os.getenv("DATA_SOURCE", "synthetic"),
        coinbase_api_key=os.getenv("COINBASE_API_KEY"),
        coinbase_api_secret=os.getenv("COINBASE_API_SECRET"),
        max_requests_per_sec=int(os.getenv("MAX_REQUESTS_PER_SECOND", "8")),
        alt_exposure_usd_cap=float(os.getenv("ALT_EXPOSURE_USD_CAP", "1500")),
        reset_state_on_start=os.getenv("RESET_STATE_ON_START", "true").lower() == "true",
        tuning_enabled=os.getenv("TUNING_ENABLED", "true").lower() == "true",
        activity_target_tph=int(os.getenv("ACTIVITY_TARGET_TPH", "60")),
        ledger_flush_sec=int(os.getenv("LEDGER_FLUSH_SEC", "10")),
        enable_mean_reversion=os.getenv("ENABLE_MEAN_REVERSION", "true").lower() == "true",
        enable_breakout=os.getenv("ENABLE_BREAKOUT", "true").lower() == "true",
        enable_grid_overlay=os.getenv("ENABLE_GRID_OVERLAY", "false").lower() == "true",
    )


