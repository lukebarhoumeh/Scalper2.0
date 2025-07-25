"""
strategy_engine_production.py - Elite Production HFT Strategy Engine
Features:
- Multi-indicator fusion (RSI, MACD, Bollinger Bands, Volume)
- Market regime detection and adaptation
- Parallel processing for 15+ coins
- Smart signal aggregation
- Real-time performance tracking
"""

from __future__ import annotations
import threading
import signal
import time
import logging
import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# Technical indicators
import ta

# Import from production config
try:
    from config_production_hft import *
except ImportError:
    from config import *

from trade_executor_production import ProductionTradeExecutor, get_trade_executor
from market_data import get_best_bid_ask, realised_volatility
from incremental_candles import get_incremental_candles
from realtime_volatility import update_volatility, get_current_volatility
from risk_manager import check_trade_risk

logger = logging.getLogger(__name__)

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Enhanced trading signal with multi-indicator support"""
    product_id: str
    action: Action
    confidence: float  # 0-1
    size_usd: float
    indicators: Dict[str, float]  # RSI, MACD, etc.
    volatility_regime: str
    strategy_source: str
    timestamp: datetime
    risk_score: float

@dataclass
class MarketSnapshot:
    """Comprehensive market data snapshot"""
    product_id: str
    bid: float
    ask: float
    last: float
    spread_bps: float
    candles: pd.DataFrame
    volume_24h: float
    volatility_pct: float
    volatility_regime: str
    rsi: float
    macd_signal: float
    bb_position: float  # Position within Bollinger Bands (0-100)
    volume_ratio: float  # Current vs average volume
    timestamp: float

class ProductionStrategy:
    """Base class for production strategies"""
    
    name: str = "BaseStrategy"
    
    def __init__(self, executor: ProductionTradeExecutor):
        self._executor = executor
        self._cooldowns = defaultdict(float)
        self._performance = {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0,
            'sharpe': 0.0
        }
    
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """Evaluate market conditions and generate signal"""
        raise NotImplementedError
    
    def _check_cooldown(self, product_id: str) -> bool:
        """Check if we're in cooldown period"""
        last_trade = self._cooldowns.get(product_id, 0)
        return (time.time() - last_trade) >= TIME_CONTROLS["cooldown_sec"]
    
    def _update_cooldown(self, product_id: str):
        """Update cooldown timer"""
        self._cooldowns[product_id] = time.time()

class MultiIndicatorScalper(ProductionStrategy):
    """
    Advanced scalping strategy using multiple indicators
    Based on best practices from HFT research
    """
    
    name = "MultiIndicatorScalper"
    
    def __init__(self, executor: ProductionTradeExecutor):
        super().__init__(executor)
        self.config = SCALPER_CONFIG
    
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """Evaluate using RSI, MACD, and Bollinger Bands"""
        
        # Check cooldown
        if not self._check_cooldown(snapshot.product_id):
            return None
        
        # Skip if volatility too low or too high
        if snapshot.volatility_pct < VOLATILITY_BANDS["low"]["max"]:
            return None
        if snapshot.volatility_pct > VOLATILITY_BANDS["extreme"]["min"]:
            return None
        
        # Skip if spread too wide
        if snapshot.spread_bps > EXECUTION_LIMITS["max_spread_bps"]:
            return None
        
        # Calculate composite score
        buy_score = 0.0
        sell_score = 0.0
        
        # RSI signal (40% weight)
        if snapshot.rsi < self.config["rsi_oversold"]:
            buy_score += 0.4 * (self.config["rsi_oversold"] - snapshot.rsi) / self.config["rsi_oversold"]
        elif snapshot.rsi > self.config["rsi_overbought"]:
            sell_score += 0.4 * (snapshot.rsi - self.config["rsi_overbought"]) / (100 - self.config["rsi_overbought"])
        
        # Bollinger Bands signal (30% weight)
        if snapshot.bb_position < 20:  # Near lower band
            buy_score += 0.3 * (20 - snapshot.bb_position) / 20
        elif snapshot.bb_position > 80:  # Near upper band
            sell_score += 0.3 * (snapshot.bb_position - 80) / 20
        
        # MACD signal (20% weight)
        if snapshot.macd_signal > 0:
            buy_score += 0.2
        else:
            sell_score += 0.2
        
        # Volume confirmation (10% weight)
        if snapshot.volume_ratio > 1.2:  # High volume
            if buy_score > sell_score:
                buy_score += 0.1
            else:
                sell_score += 0.1
        
        # Determine action
        confidence_threshold = 0.6
        
        if buy_score > confidence_threshold:
            # Additional inventory check
            current_position = self._executor.position_usd(snapshot.product_id)
            if current_position < PER_COIN_POSITION_LIMIT:
                size = self._calculate_position_size(snapshot, buy_score)
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.BUY,
                    confidence=buy_score,
                    size_usd=size,
                    indicators={
                        'rsi': snapshot.rsi,
                        'macd': snapshot.macd_signal,
                        'bb_position': snapshot.bb_position
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
        
        elif sell_score > confidence_threshold:
            # Check if we have position to sell
            current_position = self._executor.position_base(snapshot.product_id)
            if current_position > 0:
                size = min(
                    self._calculate_position_size(snapshot, sell_score),
                    current_position * snapshot.last  # Don't sell more than we have
                )
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.SELL,
                    confidence=sell_score,
                    size_usd=size,
                    indicators={
                        'rsi': snapshot.rsi,
                        'macd': snapshot.macd_signal,
                        'bb_position': snapshot.bb_position
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
        
        return None
    
    def _calculate_position_size(self, snapshot: MarketSnapshot, confidence: float) -> float:
        """Dynamic position sizing based on volatility and confidence"""
        base_size = BASE_TRADE_SIZE_USD
        
        # Volatility adjustment
        vol_band = self._get_volatility_band(snapshot.volatility_pct)
        vol_mult = VOLATILITY_BANDS[vol_band]["size_mult"]
        
        # Confidence adjustment
        conf_mult = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        
        # Final size
        size = base_size * vol_mult * conf_mult
        
        # Apply limits
        size = max(MIN_TRADE_SIZE_USD, min(size, MAX_TRADE_SIZE_USD))
        
        return round(size, 2)
    
    def _get_volatility_band(self, volatility: float) -> str:
        """Determine volatility regime"""
        for band_name, band_config in VOLATILITY_BANDS.items():
            if band_config["min"] <= volatility < band_config["max"]:
                return band_name
        return "normal"
    
    def _calculate_risk_score(self, snapshot: MarketSnapshot) -> float:
        """Calculate risk score for the trade"""
        risk_score = 0.0
        
        # Spread risk
        risk_score += min(snapshot.spread_bps / 100, 0.3)
        
        # Volatility risk
        if snapshot.volatility_regime == "extreme":
            risk_score += 0.3
        elif snapshot.volatility_regime == "high":
            risk_score += 0.2
        
        # Time of day risk
        hour_utc = datetime.now(timezone.utc).hour
        if hour_utc in TIME_CONTROLS["low_volume_hours"]:
            risk_score += 0.2
        
        return min(risk_score, 1.0)

class MomentumBreakoutStrategy(ProductionStrategy):
    """
    Breakout strategy for trending markets
    Uses ATR and volume for confirmation
    """
    
    name = "MomentumBreakout"
    
    def __init__(self, executor: ProductionTradeExecutor):
        super().__init__(executor)
        self.config = BREAKOUT_CONFIG
    
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """Evaluate for breakout opportunities"""
        
        if not self._check_cooldown(snapshot.product_id):
            return None
        
        # Need sufficient candle data
        if len(snapshot.candles) < self.config["lookback"]:
            return None
        
        # Calculate indicators
        recent_candles = snapshot.candles.tail(self.config["lookback"])
        
        # Price levels
        resistance = recent_candles['high'].max()
        support = recent_candles['low'].min()
        current_price = snapshot.last
        
        # ATR for volatility
        atr = self._calculate_atr(recent_candles, self.config["atr_window"])
        
        # Volume analysis
        avg_volume = recent_candles['volume'].mean()
        current_volume = recent_candles['volume'].iloc[-1]
        volume_spike = current_volume > avg_volume * self.config["volume_multiplier"]
        
        # Breakout detection
        breakout_threshold = atr * self.config["atr_multiplier"]
        
        # Bullish breakout
        if current_price > resistance - breakout_threshold and volume_spike:
            if self._executor.position_usd(snapshot.product_id) < PER_COIN_POSITION_LIMIT:
                confidence = min(0.8, 0.5 + (current_volume / avg_volume - 1) * 0.1)
                size = self._calculate_position_size(snapshot, confidence)
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.BUY,
                    confidence=confidence,
                    size_usd=size,
                    indicators={
                        'resistance': resistance,
                        'atr': atr,
                        'volume_ratio': current_volume / avg_volume
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
        
        # Bearish breakdown
        elif current_price < support + breakout_threshold and volume_spike:
            current_position = self._executor.position_base(snapshot.product_id)
            if current_position > 0:
                confidence = min(0.8, 0.5 + (current_volume / avg_volume - 1) * 0.1)
                size = min(
                    self._calculate_position_size(snapshot, confidence),
                    current_position * snapshot.last
                )
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.SELL,
                    confidence=confidence,
                    size_usd=size,
                    indicators={
                        'support': support,
                        'atr': atr,
                        'volume_ratio': current_volume / avg_volume
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
        
        return None
    
    def _calculate_atr(self, candles: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        high = candles['high']
        low = candles['low']
        close = candles['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _calculate_position_size(self, snapshot: MarketSnapshot, confidence: float) -> float:
        """Dynamic position sizing based on volatility and confidence"""
        base_size = BASE_TRADE_SIZE_USD
        
        # Volatility adjustment
        vol_band = self._get_volatility_band(snapshot.volatility_pct)
        vol_mult = VOLATILITY_BANDS[vol_band]["size_mult"]
        
        # Confidence adjustment
        conf_mult = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        
        # Final size
        size = base_size * vol_mult * conf_mult
        
        # Apply limits
        size = max(MIN_TRADE_SIZE_USD, min(size, MAX_TRADE_SIZE_USD))
        
        return round(size, 2)
    
    def _get_volatility_band(self, volatility: float) -> str:
        """Determine volatility regime"""
        for band_name, band_config in VOLATILITY_BANDS.items():
            if band_config["min"] <= volatility < band_config["max"]:
                return band_name
        return "normal"
    
    def _calculate_risk_score(self, snapshot: MarketSnapshot) -> float:
        """Calculate risk score for the trade"""
        risk_score = 0.0
        
        # Spread risk
        risk_score += min(snapshot.spread_bps / 100, 0.3)
        
        # Volatility risk
        if snapshot.volatility_regime == "extreme":
            risk_score += 0.3
        elif snapshot.volatility_regime == "high":
            risk_score += 0.2
        
        # Time of day risk
        hour_utc = datetime.now(timezone.utc).hour
        if hour_utc in TIME_CONTROLS["low_volume_hours"]:
            risk_score += 0.2
        
        return min(risk_score, 1.0)

class ProductionStrategyEngine:
    """
    Production-grade strategy engine with parallel processing
    """
    
    def __init__(self, executor: ProductionTradeExecutor):
        self._executor = executor
        self._strategies: List[ProductionStrategy] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self._tick_count = 0
        self._signal_count = 0
        self._execution_count = 0
        self._error_count = 0
        self._last_health_check = time.time()
        self._start_time = time.time()  # Track when engine was created
        
        # Parallel processing
        self._executor_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(len(TRADE_COINS), 10),
            thread_name_prefix="Strategy-Worker"
        )
        
        logger.info("Production Strategy Engine initialized")
    
    def register(self, strategy: ProductionStrategy):
        """Register a trading strategy"""
        self._strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")
    
    def start(self):
        """Start the strategy engine"""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Engine already running")
        
        logger.info(f"Starting Production Strategy Engine with {len(TRADE_COINS)} coins")
        self._stop_event.clear()
        
        # Register default strategies
        if not self._strategies:
            self.register(MultiIndicatorScalper(self._executor))
            self.register(MomentumBreakoutStrategy(self._executor))
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, lambda *_: self.stop())
    
    def stop(self):
        """Stop the strategy engine"""
        logger.info("Stopping Production Strategy Engine...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=10)
        
        self._executor_pool.shutdown(wait=True)
        
        # Print final stats
        self._print_performance_summary()
    
    def _run_loop(self):
        """Main execution loop"""
        logger.info("Strategy engine started")
        
        while not self._stop_event.is_set():
            try:
                loop_start = time.time()
                
                # Execute tick
                self._tick()
                
                # Dynamic sleep based on market activity
                sleep_time = self._calculate_sleep_time()
                
                # Sleep with periodic wake-ups
                sleep_end = time.time() + sleep_time
                while time.time() < sleep_end and not self._stop_event.is_set():
                    time.sleep(0.1)
                
                # Health check
                if time.time() - self._last_health_check > 60:
                    self._perform_health_check()
                    self._last_health_check = time.time()
                
            except Exception as e:
                logger.error(f"Strategy loop error: {e}", exc_info=True)
                self._error_count += 1
                time.sleep(5)
    
    def _tick(self):
        """Execute one strategy evaluation cycle"""
        self._tick_count += 1
        
        # Collect snapshots in parallel
        snapshot_futures = {}
        
        for product_id in [f"{coin}-USD" for coin in TRADE_COINS]:
            future = self._executor_pool.submit(self._collect_snapshot, product_id)
            snapshot_futures[product_id] = future
        
        # Process results as they complete
        snapshots = {}
        for product_id, future in snapshot_futures.items():
            try:
                snapshot = future.result(timeout=5.0)
                if snapshot:
                    snapshots[product_id] = snapshot
            except Exception as e:
                logger.warning(f"Failed to get snapshot for {product_id}: {e}")
        
        # Evaluate strategies on snapshots
        if snapshots:
            self._evaluate_strategies(snapshots)
    
    def _collect_snapshot(self, product_id: str) -> Optional[MarketSnapshot]:
        """Collect comprehensive market snapshot"""
        try:
            # Get market data
            bid, ask = get_best_bid_ask(product_id, max_staleness_seconds=2.0)
            
            # Skip if stale or invalid
            if bid <= 0 or ask <= 0:
                return None
            
            # Get candles
            candles = get_incremental_candles(product_id, window_size=100)
            if candles.empty or len(candles) < 50:
                return None
            
            # Calculate indicators
            close_prices = candles['close'].values
            high_prices = candles['high'].values
            low_prices = candles['low'].values
            volumes = candles['volume'].values
            
            # RSI
            rsi = ta.momentum.RSIIndicator(close=pd.Series(close_prices), window=14).rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(close=pd.Series(close_prices))
            macd_signal = macd.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=pd.Series(close_prices), window=20, window_dev=2)
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            bb_position = ((close_prices[-1] - bb_low) / (bb_high - bb_low)) * 100 if bb_high > bb_low else 50
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            volatility = realised_volatility(candles)
            vol_regime = self._classify_volatility_regime(volatility)
            
            # Update real-time volatility tracker
            update_volatility(product_id, close_prices[-1])
            
            return MarketSnapshot(
                product_id=product_id,
                bid=bid,
                ask=ask,
                last=(bid + ask) / 2,
                spread_bps=((ask - bid) / bid) * 10000,
                candles=candles,
                volume_24h=np.sum(volumes),
                volatility_pct=volatility,
                volatility_regime=vol_regime,
                rsi=rsi,
                macd_signal=macd_signal,
                bb_position=bb_position,
                volume_ratio=volume_ratio,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.debug(f"Snapshot collection failed for {product_id}: {e}")
            return None
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility into regime"""
        for regime, bounds in VOLATILITY_BANDS.items():
            if bounds["min"] <= volatility < bounds["max"]:
                return regime
        return "normal"
    
    def _evaluate_strategies(self, snapshots: Dict[str, MarketSnapshot]):
        """Evaluate all strategies on market snapshots"""
        
        # Evaluate each strategy on each snapshot
        all_signals = []
        
        for strategy in self._strategies:
            for product_id, snapshot in snapshots.items():
                signal = strategy.evaluate(snapshot)
                if signal:
                    all_signals.append(signal)
        
        # Process signals with fusion logic
        self._process_signals(all_signals)
    
    def _process_signals(self, signals: List[TradingSignal]):
        """Process signals with conflict resolution"""
        
        # Group signals by product
        product_signals = defaultdict(list)
        for signal in signals:
            product_signals[signal.product_id].append(signal)
        
        # Process each product
        for product_id, prod_signals in product_signals.items():
            # If multiple signals, aggregate
            if len(prod_signals) > 1:
                final_signal = self._aggregate_signals(prod_signals)
            else:
                final_signal = prod_signals[0]
            
            # Execute if confident enough
            if final_signal and final_signal.confidence >= 0.6:
                self._execute_signal(final_signal)
    
    def _aggregate_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Aggregate multiple signals for same product"""
        
        # Separate by action
        buy_signals = [s for s in signals if s.action == Action.BUY]
        sell_signals = [s for s in signals if s.action == Action.SELL]
        
        # If conflicting, skip
        if buy_signals and sell_signals:
            logger.info(f"Conflicting signals for {signals[0].product_id}, skipping")
            return None
        
        # Average the signals
        if buy_signals:
            signals_to_avg = buy_signals
        else:
            signals_to_avg = sell_signals
        
        # Calculate weighted average
        total_weight = sum(s.confidence for s in signals_to_avg)
        avg_confidence = total_weight / len(signals_to_avg)
        avg_size = sum(s.size_usd * s.confidence for s in signals_to_avg) / total_weight
        
        # Return strongest signal with averaged values
        strongest = max(signals_to_avg, key=lambda s: s.confidence)
        strongest.confidence = avg_confidence
        strongest.size_usd = avg_size
        
        return strongest
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            self._signal_count += 1
            
            # Use enhanced formatter for signal display
            from enhanced_output_formatter import EnhancedFormatter
            signal_data = {
                'action': signal.action.value,
                'product_id': signal.product_id,
                'size_usd': signal.size_usd,
                'strategy': signal.strategy_source,
                'confidence': signal.confidence,
                'risk_score': signal.risk_score
            }
            EnhancedFormatter.print_signal(signal_data)
            
            # Execute trade
            if signal.action == Action.BUY:
                order_id = self._executor.market_buy(
                    signal.product_id,
                    signal.size_usd,
                    signal.strategy_source
                )
            else:  # SELL
                order_id = self._executor.market_sell(
                    signal.product_id,
                    signal.size_usd,
                    signal.strategy_source
                )
            
            if order_id:
                self._execution_count += 1
                logger.info(f"Trade executed: {order_id}")
            
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            self._error_count += 1
    
    def _calculate_sleep_time(self) -> float:
        """Dynamic sleep time based on market conditions"""
        
        # Get average volatility
        avg_volatility = np.mean([
            get_current_volatility(f"{coin}-USD", "5m") 
            for coin in TRADE_COINS[:5]  # Sample first 5
        ])
        
        # High volatility = faster polling
        if avg_volatility > 10:
            return FAST_POLL_INTERVAL_SEC
        else:
            return POLL_INTERVAL_SEC
    
    def _perform_health_check(self):
        """Perform system health check"""
        
        # Check error rate
        if self._tick_count > 0:
            error_rate = self._error_count / self._tick_count
            if error_rate > 0.1:  # >10% errors
                logger.warning(f"High error rate: {error_rate:.1%}")
        
        # Check execution rate
        if self._signal_count > 0:
            execution_rate = self._execution_count / self._signal_count
            logger.info(f"Execution rate: {execution_rate:.1%}")
        
        # Log stats
        stats = self._executor.get_daily_stats()
        logger.info(f"Daily stats: PnL=${stats['daily_pnl']:.2f}, Trades={stats['total_trades']}")
    
    def is_healthy(self) -> bool:
        """Check if strategy engine is healthy and operational"""
        try:
            # Check if engine is running
            if not self._thread or not self._thread.is_alive():
                return False
            
            # Allow startup grace period (first 30 seconds)
            time_since_start = time.time() - (getattr(self, '_start_time', time.time() - 31))
            if time_since_start < 30:
                # During startup, just check if thread is alive
                return self._thread.is_alive()
            
            # Check error rate after startup
            if self._tick_count > 10:  # After some initial ticks
                error_rate = self._error_count / self._tick_count
                if error_rate > 0.2:  # >20% error rate is unhealthy
                    return False
            
            # Check if we've had recent activity
            time_since_health_check = time.time() - self._last_health_check
            if time_since_health_check > 300:  # No health check in 5 minutes
                return False
            
            return True
        except:
            return False
    
    def _print_performance_summary(self):
        """Print performance summary on shutdown"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total ticks: {self._tick_count}")
        logger.info(f"Signals generated: {self._signal_count}")
        logger.info(f"Trades executed: {self._execution_count}")
        logger.info(f"Error count: {self._error_count}")
        
        # Get final stats from executor
        stats = self._executor.get_daily_stats()
        logger.info(f"Daily PnL: ${stats['daily_pnl']:.2f}")
        logger.info(f"Win rate: {stats['win_rate']:.1%}")
        logger.info(f"Avg slippage: {stats['avg_slippage_bps']:.1f} bps")
        logger.info("=" * 60) 