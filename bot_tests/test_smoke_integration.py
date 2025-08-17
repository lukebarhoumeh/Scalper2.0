import asyncio
import time

from bot.config import load_config, BotConfig
from bot.app import App


async def _run_short(app: App, seconds: float):
    t0 = time.time()
    async def stopper():
        while time.time() - t0 < seconds:
            await asyncio.sleep(0.2)
        app.request_shutdown()

    stopper_task = asyncio.create_task(stopper())
    try:
        await app.run()
    finally:
        stopper_task.cancel()


def test_smoke_runs_for_30s_event_loop():
    cfg = load_config()
    cfg = type(cfg)(
        trading_pairs=["BTC-USD"],
        paper_trading=True,
        base_position_usd=cfg.base_position_usd,
        max_position_usd=cfg.max_position_usd,
        max_total_exposure_usd=cfg.max_total_exposure_usd,
        starting_cash_usd=10000.0,
        rsi_buy=cfg.rsi_buy,
        rsi_sell=cfg.rsi_sell,
        sma_fast=cfg.sma_fast,
        sma_slow=cfg.sma_slow,
        spread_bps_max=cfg.spread_bps_max,
        risk_per_trade=cfg.risk_per_trade,
        max_daily_loss_percent=cfg.max_daily_loss_percent,
        circuit_breaker_errors=cfg.circuit_breaker_errors,
        poll_interval_sec=0.2,
        ui_refresh_hz=10.0,
        state_path="bot_state_test.json",
        logs_dir="logs",
        take_profit_bps=cfg.take_profit_bps,
        stop_loss_bps=cfg.stop_loss_bps,
        trailing_stop_bps=cfg.trailing_stop_bps,
        data_source="synthetic",
        coinbase_api_key=None,
        coinbase_api_secret=None,
        max_requests_per_sec=cfg.max_requests_per_sec,
        alt_exposure_usd_cap=cfg.alt_exposure_usd_cap,
    )
    app = App(cfg)

    asyncio.run(_run_short(app, 5.0))

    # at least one price processed
    assert app.broker.last_prices.get("BTC-USD") is not None
    # UI would have rendered; ensure no crash (implicit by run completion)
    # Trade pipeline ability: try to place an order explicitly
    price = app.broker.last_prices["BTC-USD"]
    qty = 10.0 / price
    from bot.models import Order, OrderType, Side

    fill = app.broker.place_order(Order(symbol="BTC-USD", side=Side.BUY, qty=qty, type=OrderType.MARKET))
    assert fill is not None


