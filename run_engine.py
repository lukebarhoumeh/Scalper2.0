# run_engine.py – entry point
from trade_logger import TradeLogger
from trade_executor import TradeExecutor
from strategy_engine import StrategyEngine, ScalperStrategy, BreakoutStrategy
from config import POLL_INTERVAL_SEC

if __name__ == "__main__":
    logger = TradeLogger()
    exe    = TradeExecutor(logger)

    engine = StrategyEngine(exe, poll_interval=POLL_INTERVAL_SEC)
    engine.register(ScalperStrategy(exe))
    engine.register(BreakoutStrategy(exe))

    engine.start()
    engine._thread.join()                 # wait until Ctrl‑C or stop()

    # print realised P n L summary
    print("\n───────── REALISED P n L ─────────")
    for pid, pnl in exe.pnl_breakdown().items():
        print(f"{pid:<8}  {pnl:>10.2f} USD")
    print("───────────────────────────────────")
    print(f"TOTAL     {exe.running_pnl():>10.2f} USD\n")
