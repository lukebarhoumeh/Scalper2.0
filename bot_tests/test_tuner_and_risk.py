from bot.dynamic_config import DynamicConfigManager, StrategyParams


def test_tuner_bounds_and_activity():
    params = StrategyParams(rsi_buy=45, rsi_sell=55, sma_fast=8, sma_slow=21, max_spread_bps=150, take_profit_bps=35, stop_loss_bps=45, trailing_stop_bps=30)
    dyn = DynamicConfigManager(initial=params, history_path="logs/param_history_test.json", cooldown_updates=1, activity_target_tph=60)
    # Below target activity and negative markout -> should loosen gates within bounds
    changed = dyn.maybe_update(session_win_rate_pct=40, session_sharpe=-0.2, avg_markout_5s_bps=-3.0, avg_spread_bps=130, avg_vol_pct=1.0, activity_tph=10)
    assert changed
    p = dyn.get_params()
    assert 20 <= p.rsi_buy <= 70
    assert 30 <= p.rsi_sell <= 80
    assert 50 <= p.max_spread_bps <= 300


