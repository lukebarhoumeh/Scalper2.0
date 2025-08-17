# health_check_market.py
from config import TRADE_COINS
from market_data import get_best_bid_ask

def normalize(pid: str) -> str:
    return pid if "-" in pid else f"{pid}-USD"

if __name__ == "__main__":
    for raw in TRADE_COINS:
        product_id = normalize(raw)
        bid, ask = get_best_bid_ask(product_id)
        print(f"{product_id:<10} | bid={bid:,.6f}  ask={ask:,.6f}")
