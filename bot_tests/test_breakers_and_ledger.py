from datetime import datetime, timezone, timedelta
from bot.state import initialize_account
from bot.broker import PaperBroker
from bot.models import Order, OrderType, Side
from bot.ledger import Ledger


def test_ledger_flush_incremental(tmp_path):
    acct = initialize_account(1000.0)
    brk = PaperBroker(acct)
    brk.update_price("BTC-USD", bid=100.0, ask=100.2)
    o = Order(symbol="BTC-USD", side=Side.BUY, qty=0.00001, type=OrderType.MARKET)
    f = brk.place_order(o)
    led = Ledger()
    led.add_fill(o, f, 100.1, 100.1, acct.realized_pnl)
    p = tmp_path / "ledger.csv"
    led.flush_to_csv(str(p))
    # No new entries after second flush
    led.flush_to_csv(str(p))
    assert p.exists() and p.read_text().count("BTC-USD") == 1


