from bot.models import Order, OrderType, Side, Position, Fill


def test_order_has_side_price_qty():
    o = Order(symbol="BTC-USD", side=Side.BUY, qty=0.01, type=OrderType.MARKET)
    assert o.side == Side.BUY
    assert o.qty == 0.01
    assert o.type == OrderType.MARKET


def test_position_apply_fill_roundtrip():
    pos = Position(symbol="BTC-USD")
    buy = Fill(order_id="1", symbol="BTC-USD", side=Side.BUY, price=100.0, qty=1.0)
    pos.apply_fill(buy)
    assert pos.qty == 1.0 and pos.avg_price == 100.0
    sell = Fill(order_id="2", symbol="BTC-USD", side=Side.SELL, price=110.0, qty=1.0)
    realized = pos.apply_fill(sell)
    assert realized == 10.0
    assert pos.qty == 0.0 and pos.avg_price == 0.0


