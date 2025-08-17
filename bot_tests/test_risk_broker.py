from bot.state import initialize_account
from bot.broker import PaperBroker
from bot.models import Order, OrderType, Side
from bot.risk import RiskLimits, compute_position_size_usd, can_increase_exposure


def test_paper_broker_fill_and_equity():
    acct = initialize_account(1000.0)
    brk = PaperBroker(acct)
    brk.update_price("BTC-USD", bid=100.0, ask=100.2)
    usd = 100.0
    qty = usd / 100.1
    o = Order(symbol="BTC-USD", side=Side.BUY, qty=qty, type=OrderType.MARKET)
    f = brk.place_order(o)
    assert f is not None
    assert acct.cash < 1000.0 and acct.equity == acct.cash + acct.positions["BTC-USD"].qty * brk.last_prices["BTC-USD"]


def test_risk_calcs():
    acct = initialize_account(10000.0)
    limits = RiskLimits(risk_per_trade=0.01, max_total_exposure_usd=2000.0, max_daily_loss_percent=0.1)
    usd = compute_position_size_usd(acct, limits)
    assert usd == 100.0
    assert can_increase_exposure(acct, {"BTC-USD": 100.0}, limits, 100.0)


