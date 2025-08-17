"""Machine Learning Predictor for Trading Signals

This module implements ML-based prediction models for generating
high-confidence trading signals using market microstructure patterns.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
import joblib
from datetime import datetime, timezone
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MLSignal:
    """ML-generated trading signal"""
    symbol: str
    direction: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    predicted_move_bps: float
    features_importance: Dict[str, float]
    model_name: str
    timestamp: datetime


@dataclass
class MarketFeatures:
    """Features for ML prediction"""
    # Price features
    price_roc_1m: float
    price_roc_5m: float
    price_roc_15m: float
    price_position_in_range: float  # 0-1 position in daily range
    
    # Volume features
    volume_ratio_1m: float  # Current vs average
    volume_ratio_5m: float
    buy_sell_ratio: float
    
    # Technical features
    rsi: float
    rsi_slope: float
    macd_histogram: float
    bb_position: float  # Position in Bollinger Bands
    
    # Microstructure features
    spread_bps: float
    spread_ma_ratio: float  # Current spread vs MA
    bid_ask_imbalance: float
    trade_intensity: float  # Trades per second
    
    # Order flow features
    order_flow_imbalance: float
    large_trade_ratio: float
    aggressive_buy_ratio: float
    
    # Time features
    hour_of_day: int
    day_of_week: int
    minutes_since_open: int


class MLPredictor:
    """
    Machine Learning predictor for trading signals.
    
    Uses ensemble methods to predict:
    - Direction (classification)
    - Magnitude (regression)
    - Optimal holding period
    """
    
    def __init__(self, 
                 min_confidence: float = 0.65,
                 lookback_minutes: int = 60,
                 update_frequency_minutes: int = 30):
        
        self.min_confidence = min_confidence
        self.lookback_minutes = lookback_minutes
        self.update_frequency_minutes = update_frequency_minutes
        
        # Models
        self.direction_model: Optional[RandomForestClassifier] = None
        self.magnitude_model: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        
        # Feature tracking
        self.feature_history: Dict[str, deque] = {}
        self.prediction_history: List[MLSignal] = []
        self.last_update: Optional[datetime] = None
        
        # Model performance tracking
        self.model_metrics = {
            "direction_accuracy": 0.0,
            "magnitude_mae": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Load pre-trained models if available
        self._load_models()
    
    def extract_features(self, symbol: str, price_history: List[float], 
                        volume_history: List[float], order_book: Dict,
                        recent_trades: List[Dict]) -> MarketFeatures:
        """Extract features from market data"""
        
        if len(price_history) < 60:
            # Return default features if insufficient history
            return self._get_default_features()
        
        # Price features
        current_price = price_history[-1]
        price_1m_ago = price_history[-2] if len(price_history) > 1 else current_price
        price_5m_ago = price_history[-10] if len(price_history) > 10 else current_price
        price_15m_ago = price_history[-30] if len(price_history) > 30 else current_price
        
        price_roc_1m = ((current_price - price_1m_ago) / price_1m_ago) * 10000
        price_roc_5m = ((current_price - price_5m_ago) / price_5m_ago) * 10000
        price_roc_15m = ((current_price - price_15m_ago) / price_15m_ago) * 10000
        
        # Daily range position
        daily_high = max(price_history[-1440:]) if len(price_history) > 1440 else max(price_history)
        daily_low = min(price_history[-1440:]) if len(price_history) > 1440 else min(price_history)
        price_position = (current_price - daily_low) / (daily_high - daily_low) if daily_high > daily_low else 0.5
        
        # Volume features
        current_volume = volume_history[-1] if volume_history else 100
        avg_volume_1m = np.mean(volume_history[-2:]) if len(volume_history) > 2 else current_volume
        avg_volume_5m = np.mean(volume_history[-10:]) if len(volume_history) > 10 else current_volume
        
        volume_ratio_1m = current_volume / avg_volume_1m if avg_volume_1m > 0 else 1.0
        volume_ratio_5m = current_volume / avg_volume_5m if avg_volume_5m > 0 else 1.0
        
        # Order book features
        bid = order_book.get('bid', current_price)
        ask = order_book.get('ask', current_price)
        bid_size = order_book.get('bid_size', 1.0)
        ask_size = order_book.get('ask_size', 1.0)
        
        spread_bps = ((ask - bid) / current_price) * 10000
        bid_ask_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0
        
        # Trade flow features
        buy_volume = sum(t['size'] for t in recent_trades if t.get('side') == 'buy')
        sell_volume = sum(t['size'] for t in recent_trades if t.get('side') == 'sell')
        total_volume = buy_volume + sell_volume
        
        buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 2.0
        trade_intensity = len(recent_trades) / 60.0  # Trades per second
        
        # Large trade detection
        avg_trade_size = total_volume / len(recent_trades) if recent_trades else 1.0
        large_trades = [t for t in recent_trades if t['size'] > avg_trade_size * 2]
        large_trade_ratio = len(large_trades) / len(recent_trades) if recent_trades else 0
        
        # Aggressive orders (market orders)
        aggressive_buys = sum(1 for t in recent_trades if t.get('aggressive') and t.get('side') == 'buy')
        aggressive_buy_ratio = aggressive_buys / len(recent_trades) if recent_trades else 0.5
        
        # Technical indicators (simplified)
        rsi = self._calculate_rsi(price_history, 14)
        rsi_slope = self._calculate_rsi_slope(price_history)
        macd_hist = self._calculate_macd_histogram(price_history)
        bb_position = self._calculate_bb_position(price_history)
        
        # Time features
        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()
        minutes_since_open = (now.hour * 60 + now.minute) % 1440
        
        # Order flow imbalance (simplified)
        order_flow_imbalance = buy_sell_ratio - 1.0
        
        # Spread MA ratio
        spread_ma = np.mean([((order_book.get('ask', p) - order_book.get('bid', p)) / p) * 10000 
                            for p in price_history[-10:]])
        spread_ma_ratio = spread_bps / spread_ma if spread_ma > 0 else 1.0
        
        return MarketFeatures(
            price_roc_1m=price_roc_1m,
            price_roc_5m=price_roc_5m,
            price_roc_15m=price_roc_15m,
            price_position_in_range=price_position,
            volume_ratio_1m=volume_ratio_1m,
            volume_ratio_5m=volume_ratio_5m,
            buy_sell_ratio=buy_sell_ratio,
            rsi=rsi,
            rsi_slope=rsi_slope,
            macd_histogram=macd_hist,
            bb_position=bb_position,
            spread_bps=spread_bps,
            spread_ma_ratio=spread_ma_ratio,
            bid_ask_imbalance=bid_ask_imbalance,
            trade_intensity=trade_intensity,
            order_flow_imbalance=order_flow_imbalance,
            large_trade_ratio=large_trade_ratio,
            aggressive_buy_ratio=aggressive_buy_ratio,
            hour_of_day=hour,
            day_of_week=day_of_week,
            minutes_since_open=minutes_since_open
        )
    
    def predict(self, symbol: str, features: MarketFeatures) -> Optional[MLSignal]:
        """Generate ML prediction"""
        
        if not self.direction_model or not self.magnitude_model:
            # Train models if not available
            if not self._train_models():
                return None
        
        # Convert features to array
        feature_array = self._features_to_array(features)
        feature_scaled = self.scaler.transform([feature_array])
        
        # Predict direction and probability
        direction_proba = self.direction_model.predict_proba(feature_scaled)[0]
        direction_classes = self.direction_model.classes_
        
        # Get best direction and confidence
        best_idx = np.argmax(direction_proba)
        direction = direction_classes[best_idx]
        confidence = direction_proba[best_idx]
        
        # Skip if confidence too low
        if confidence < self.min_confidence:
            return None
        
        # Skip if predicting "hold"
        if direction == "hold":
            return None
        
        # Predict magnitude
        predicted_move = self.magnitude_model.predict(feature_scaled)[0]
        
        # Get feature importance
        feature_importance = self._get_feature_importance()
        
        # Create signal
        signal = MLSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            predicted_move_bps=predicted_move,
            features_importance=feature_importance,
            model_name="ensemble_rf_gb",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Track prediction
        self.prediction_history.append(signal)
        
        return signal
    
    def update_models(self, market_data: List[Dict], outcomes: List[Dict]) -> bool:
        """Update models with new data"""
        
        # Check if update needed
        now = datetime.now(timezone.utc)
        if self.last_update:
            minutes_since_update = (now - self.last_update).total_seconds() / 60
            if minutes_since_update < self.update_frequency_minutes:
                return False
        
        # Prepare training data
        X, y_direction, y_magnitude = self._prepare_training_data(market_data, outcomes)
        
        if len(X) < 100:
            return False
        
        # Split data
        X_train, X_test, y_dir_train, y_dir_test, y_mag_train, y_mag_test = train_test_split(
            X, y_direction, y_magnitude, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train direction model
        self.direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.direction_model.fit(X_train_scaled, y_dir_train)
        
        # Train magnitude model
        self.magnitude_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.magnitude_model.fit(X_train_scaled, y_mag_train)
        
        # Evaluate models
        dir_accuracy = self.direction_model.score(X_test_scaled, y_dir_test)
        mag_mae = np.mean(np.abs(self.magnitude_model.predict(X_test_scaled) - y_mag_test))
        
        self.model_metrics["direction_accuracy"] = dir_accuracy
        self.model_metrics["magnitude_mae"] = mag_mae
        
        # Save models
        self._save_models()
        
        self.last_update = now
        
        return True
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get model performance metrics"""
        
        if not self.prediction_history:
            return self.model_metrics
        
        # Calculate recent performance
        recent_predictions = self.prediction_history[-100:]
        
        # This would need actual outcome data to calculate properly
        # For now, return stored metrics
        return self.model_metrics
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_slope(self, prices: List[float]) -> float:
        """Calculate RSI slope"""
        if len(prices) < 20:
            return 0.0
        
        rsi_values = []
        for i in range(5):
            end_idx = len(prices) - i
            start_idx = end_idx - 14
            if start_idx >= 0:
                rsi = self._calculate_rsi(prices[:end_idx], 14)
                rsi_values.append(rsi)
        
        if len(rsi_values) < 2:
            return 0.0
        
        # Simple slope
        return (rsi_values[0] - rsi_values[-1]) / len(rsi_values)
    
    def _calculate_macd_histogram(self, prices: List[float]) -> float:
        """Calculate MACD histogram"""
        if len(prices) < 26:
            return 0.0
        
        # Simple EMA calculation
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        
        macd = ema12 - ema26
        signal = self._ema([macd], 9)
        
        return macd - signal
    
    def _calculate_bb_position(self, prices: List[float], period: int = 20) -> float:
        """Calculate position in Bollinger Bands"""
        if len(prices) < period:
            return 0.5
        
        recent = prices[-period:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std == 0:
            return 0.5
        
        upper = mean + 2 * std
        lower = mean - 2 * std
        current = prices[-1]
        
        position = (current - lower) / (upper - lower)
        return np.clip(position, 0, 1)
    
    def _ema(self, values: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(values) < period:
            return values[-1] if values else 0
        
        multiplier = 2 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = (value - ema) * multiplier + ema
        
        return ema
    
    def _features_to_array(self, features: MarketFeatures) -> np.ndarray:
        """Convert features to numpy array"""
        return np.array([
            features.price_roc_1m,
            features.price_roc_5m,
            features.price_roc_15m,
            features.price_position_in_range,
            features.volume_ratio_1m,
            features.volume_ratio_5m,
            features.buy_sell_ratio,
            features.rsi,
            features.rsi_slope,
            features.macd_histogram,
            features.bb_position,
            features.spread_bps,
            features.spread_ma_ratio,
            features.bid_ask_imbalance,
            features.trade_intensity,
            features.order_flow_imbalance,
            features.large_trade_ratio,
            features.aggressive_buy_ratio,
            features.hour_of_day,
            features.day_of_week,
            features.minutes_since_open
        ])
    
    def _get_default_features(self) -> MarketFeatures:
        """Get default neutral features"""
        return MarketFeatures(
            price_roc_1m=0.0,
            price_roc_5m=0.0,
            price_roc_15m=0.0,
            price_position_in_range=0.5,
            volume_ratio_1m=1.0,
            volume_ratio_5m=1.0,
            buy_sell_ratio=1.0,
            rsi=50.0,
            rsi_slope=0.0,
            macd_histogram=0.0,
            bb_position=0.5,
            spread_bps=10.0,
            spread_ma_ratio=1.0,
            bid_ask_imbalance=0.0,
            trade_intensity=1.0,
            order_flow_imbalance=0.0,
            large_trade_ratio=0.1,
            aggressive_buy_ratio=0.5,
            hour_of_day=12,
            day_of_week=2,
            minutes_since_open=180
        )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from models"""
        if not self.direction_model:
            return {}
        
        feature_names = [
            "price_roc_1m", "price_roc_5m", "price_roc_15m", "price_position",
            "volume_ratio_1m", "volume_ratio_5m", "buy_sell_ratio",
            "rsi", "rsi_slope", "macd_hist", "bb_position",
            "spread_bps", "spread_ma_ratio", "bid_ask_imbalance",
            "trade_intensity", "order_flow_imbalance", "large_trade_ratio",
            "aggressive_buy_ratio", "hour", "day_of_week", "minutes_since_open"
        ]
        
        importances = self.direction_model.feature_importances_
        
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
    
    def _prepare_training_data(self, market_data: List[Dict], 
                              outcomes: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from market data and outcomes"""
        
        X = []
        y_direction = []
        y_magnitude = []
        
        # This is a simplified version - in production, you'd have
        # a more sophisticated data pipeline
        
        for data, outcome in zip(market_data, outcomes):
            # Extract features from historical data
            features = data.get('features')
            if features:
                X.append(self._features_to_array(features))
                
                # Direction: buy, sell, or hold
                price_change = outcome.get('price_change_bps', 0)
                if price_change > 5:
                    y_direction.append("buy")
                elif price_change < -5:
                    y_direction.append("sell")
                else:
                    y_direction.append("hold")
                
                # Magnitude
                y_magnitude.append(abs(price_change))
        
        return np.array(X), np.array(y_direction), np.array(y_magnitude)
    
    def _train_models(self) -> bool:
        """Train models with synthetic data for demo"""
        
        # Generate synthetic training data
        n_samples = 1000
        n_features = 21
        
        # Random features
        X = np.random.randn(n_samples, n_features)
        
        # Synthetic labels with some pattern
        y_direction = []
        y_magnitude = []
        
        for i in range(n_samples):
            # Create some correlation between features and outcome
            signal_strength = X[i, 0] * 0.3 + X[i, 1] * 0.2 + X[i, 7] * 0.1
            
            if signal_strength > 0.5:
                y_direction.append("buy")
                y_magnitude.append(np.random.uniform(10, 50))
            elif signal_strength < -0.5:
                y_direction.append("sell")
                y_magnitude.append(np.random.uniform(10, 50))
            else:
                y_direction.append("hold")
                y_magnitude.append(np.random.uniform(0, 20))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.direction_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.direction_model.fit(X_scaled, y_direction)
        
        self.magnitude_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        self.magnitude_model.fit(X_scaled, y_magnitude)
        
        return True
    
    def _save_models(self) -> None:
        """Save trained models"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # Save models
        joblib.dump(self.direction_model, model_dir / "direction_model.pkl")
        joblib.dump(self.magnitude_model, model_dir / "magnitude_model.pkl")
        joblib.dump(self.scaler, model_dir / "feature_scaler.pkl")
        
        # Save metadata
        metadata = {
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "model_metrics": self.model_metrics,
            "feature_count": 21
        }
        
        with open(model_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_models(self) -> bool:
        """Load pre-trained models"""
        model_dir = Path("models")
        
        try:
            if (model_dir / "direction_model.pkl").exists():
                self.direction_model = joblib.load(model_dir / "direction_model.pkl")
                self.magnitude_model = joblib.load(model_dir / "magnitude_model.pkl")
                self.scaler = joblib.load(model_dir / "feature_scaler.pkl")
                
                # Load metadata
                with open(model_dir / "model_metadata.json", "r") as f:
                    metadata = json.load(f)
                    self.model_metrics = metadata.get("model_metrics", {})
                    if metadata.get("last_update"):
                        self.last_update = datetime.fromisoformat(metadata["last_update"])
                
                return True
        except Exception as e:
            print(f"Failed to load models: {e}")
        
        return False
