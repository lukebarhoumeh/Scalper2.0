# health_check_trade.py
from trade_logger import TradeLogger
from trade_executor import TradeExecutor
from config import TRADE_COINS, TRADE_SIZE_USD


def normalize(pid: str) -> str:
    return pid if "-" in pid else f"{pid}-USD"

if __name__ == "__main__":
    coin = TRADE_COINS[0]
    product = f"{coin}-USD"
    logger = TradeLogger()
    exe = TradeExecutor(logger)

    print("Simulating BUY…")
    exe.market_buy(product, TRADE_SIZE_USD, "health_check")

    print("Simulating SELL…")
    exe.market_sell(product, TRADE_SIZE_USD, "health_check")

    print("Done. Inspect logs/trades.csv")
