from bot.indicators import simple_moving_average, rsi, breakout


def test_sma():
    values = list(range(1, 11))  # 1..10
    assert simple_moving_average(values, 5) == sum([6, 7, 8, 9, 10]) / 5


def test_rsi_bounds():
    uptrend = [i for i in range(1, 20)]
    v = rsi(uptrend, 14)
    assert v is not None and 0 <= v <= 100


def test_breakout():
    vals = [1] * 19 + [2]
    assert breakout(vals, 20) == "breakout_up"


