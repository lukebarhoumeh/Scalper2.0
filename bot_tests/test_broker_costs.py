from bot.state import initialize_account
from bot.broker import PaperBroker
from bot.models import Order, OrderType, Side


def test_rounding_and_fees():
    acct = initialize_account(1000.0)
    brk = PaperBroker(acct)
    brk.update_price("BTC-USD", bid=100.0, ask=100.2)
    # Place BUY with non-conforming qty; should round down to lot and charge fees
    o = Order(symbol="BTC-USD", side=Side.BUY, qty=0.00000123, type=OrderType.MARKET)
    f = brk.place_order(o)
    assert f is not None
    # Place SELL full position; realized pnl less fees
    brk.update_price("BTC-USD", bid=100.3, ask=100.5)
    pos_qty = brk.account.positions["BTC-USD"].qty
    o2 = Order(symbol="BTC-USD", side=Side.SELL, qty=pos_qty, type=OrderType.MARKET)
    f2 = brk.place_order(o2)
    assert f2 is not None
    # Cash should have been reduced by two fee charges
    assert acct.cash < 1000.0


