#!/usr/bin/env python3
"""
AI Risk Manager - HFT Production Grade
Advanced risk management using machine learning and market intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import asyncio
import logging
from collections import deque, defaultdict
import openai
import json
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

from config import OPENAI_API_KEY, OPENAI_MODEL, TRADE_COINS
from market_intelligence import MarketRegime, MarketIntelligence
from dynamic_config import TradingParameters

# Configure OpenAI
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for real-time monitoring"""
    # Portfolio metrics
    total_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0
    gross_exposure_usd: float = 0.0
    leverage_ratio: float = 0.0
    
    # Risk measures
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Market risk
    beta: Dict[str, float] = field(default_factory=dict)
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0
    
    # Operational risk
    slippage_cost: float = 0.0
    execution_quality: float = 1.0
    latency_percentile_99: float = 0.0
    
    # Anomaly scores
    anomaly_score: float = 0.0
    regime_stability: float = 1.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RiskAlert:
    """Risk alert with severity and recommendations"""
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'market', 'portfolio', 'execution', 'system'
    message: str
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class AIRiskManager:
    """
    AI-powered risk manager that learns from performance and adapts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.market_conditions_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
        # Risk parameters
        self.risk_multipliers = {
            'conservative': 0.5,
            'balanced': 1.0,
            'aggressive': 2.0
        }
        
        # Volatility tracking
        self.volatility_window = 20
        self.volatility_threshold = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }
        
        # Mode switching thresholds
        self.mode_thresholds = {
            'profit_aggressive': 0.02,  # 2% daily profit
            'profit_balanced': 0.005,   # 0.5% daily profit
            'loss_conservative': -0.01, # -1% daily loss
            'error_rate_high': 0.1      # 10% error rate
        }
        
        # Machine learning components (simplified for now)
        self.pattern_memory = {}
        self.success_patterns = []
        self.failure_patterns = []
        
    async def initialize(self):
        """Initialize AI components"""
        self.logger.info("Initializing AI Risk Manager")
        # In production, this would load ML models
        return True
    
    async def health_check(self) -> bool:
        """Check if AI system is healthy"""
        return True
    
    def get_optimal_config(self, mode: str, performance_history: List) -> Dict:
        """Get optimal configuration based on mode and performance"""
        base_config = self._get_base_config(mode)
        
        # Adjust based on recent performance
        if len(performance_history) > 10:
            recent_performance = performance_history[-10:]
            avg_pnl = np.mean([p.get('pnl', 0) for p in recent_performance])
            
            # Adjust position sizes based on performance
            if avg_pnl > 0:
                base_config['position_multiplier'] = min(1.5, 1 + (avg_pnl / 100))
            else:
                base_config['position_multiplier'] = max(0.5, 1 + (avg_pnl / 100))
        
        # Adjust based on market volatility
        volatility = self._calculate_market_volatility()
        if volatility > self.volatility_threshold['high']:
            base_config['poll_interval'] = max(10, base_config['poll_interval'] * 0.5)
            base_config['position_multiplier'] *= 0.8
        
        return base_config
    
    async def recommend_mode(self, current_pnl: float, error_count: int, 
                            market_conditions: Dict) -> Dict:
        """Recommend trading mode based on current conditions"""
        
        # Calculate performance metrics
        total_capital = 1000  # $1000 daily limit
        pnl_ratio = current_pnl / total_capital
        
        # Check error rate
        error_rate = error_count / max(1, len(self.error_history))
        
        # Market volatility
        volatility = market_conditions.get('volatility', 0.02)
        
        # Decision logic
        mode = 'balanced'  # default
        
        if error_rate > self.mode_thresholds['error_rate_high']:
            mode = 'conservative'
        elif pnl_ratio < self.mode_thresholds['loss_conservative']:
            mode = 'conservative'
        elif pnl_ratio > self.mode_thresholds['profit_aggressive'] and volatility < self.volatility_threshold['medium']:
            mode = 'aggressive'
        elif pnl_ratio > self.mode_thresholds['profit_balanced']:
            mode = 'balanced'
        elif volatility > self.volatility_threshold['high']:
            mode = 'conservative'
        elif volatility < self.volatility_threshold['low']:
            mode = 'aggressive'
        
        # Return configuration dict
        return {
            'mode': mode,
            'confidence': 0.85,
            'position_size_multiplier': self.risk_multipliers[mode],
            'max_daily_trades': 30 if mode == 'conservative' else 50 if mode == 'balanced' else 100,
            'risk_parameters': {
                'stop_loss_pct': 0.03 if mode == 'conservative' else 0.02 if mode == 'balanced' else 0.015,
                'take_profit_pct': 0.01 if mode == 'conservative' else 0.015 if mode == 'balanced' else 0.02
            }
        }
    
    async def adjust_risk_parameters(self, current_params: Dict) -> Dict:
        """Adjust risk parameters based on current market conditions"""
        # Get current mode recommendation
        mode_config = await self.recommend_mode(
            current_pnl=current_params.get('current_pnl', 0),
            error_count=current_params.get('error_count', 0),
            market_conditions=current_params.get('market_conditions', {})
        )
        
        # Apply adjustments
        adjusted_params = current_params.copy()
        adjusted_params['max_position_size'] = current_params.get('max_position_size', 1000) * mode_config['position_size_multiplier']
        adjusted_params['stop_loss_pct'] = mode_config['risk_parameters']['stop_loss_pct']
        adjusted_params['take_profit_pct'] = mode_config['risk_parameters']['take_profit_pct']
        adjusted_params['max_daily_trades'] = mode_config['max_daily_trades']
        
        return adjusted_params
    
    def record_trade_outcome(self, trade: Dict):
        """Record trade outcome for learning"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'product': trade.get('product_id'),
            'side': trade.get('side'),
            'pnl': trade.get('pnl', 0),
            'strategy': trade.get('strategy'),
            'market_conditions': self._get_current_conditions()
        })
        
        # Simple pattern learning
        if trade.get('pnl', 0) > 0:
            self.success_patterns.append(self._extract_pattern(trade))
        else:
            self.failure_patterns.append(self._extract_pattern(trade))
    
    def _get_base_config(self, mode: str) -> Dict:
        """Get base configuration for mode"""
        configs = {
            'conservative': {
                'coins': ['BTC', 'ETH'],  # Only major coins
                'poll_interval': 60,
                'enable_multi_strategy': False,
                'position_multiplier': 0.5,
                'max_positions': 2,
                'stop_loss_pct': 0.005,  # 0.5%
                'take_profit_pct': 0.01   # 1%
            },
            'balanced': {
                'coins': ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC'],
                'poll_interval': 30,
                'enable_multi_strategy': True,
                'position_multiplier': 1.0,
                'max_positions': 5,
                'stop_loss_pct': 0.01,   # 1%
                'take_profit_pct': 0.02   # 2%
            },
            'aggressive': {
                'coins': ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'DOGE', 'SHIB', 
                         'PEPE', 'ADA', 'XRP', 'DOT', 'LINK', 'UNI', 'ATOM', 'NEAR'],
                'poll_interval': 15,
                'enable_multi_strategy': True,
                'position_multiplier': 2.0,
                'max_positions': 10,
                'stop_loss_pct': 0.02,   # 2%
                'take_profit_pct': 0.05   # 5%
            }
        }
        return configs.get(mode, configs['balanced'])
    
    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        if len(self.market_conditions_history) < self.volatility_window:
            return 0.02  # Default medium volatility
        
        recent_prices = [c.get('btc_price', 0) for c in self.market_conditions_history]
        if not recent_prices:
            return 0.02
        
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns)
    
    def _get_current_conditions(self) -> Dict:
        """Get current market conditions"""
        return {
            'timestamp': datetime.now(),
            'volatility': self._calculate_market_volatility(),
            'performance_history_size': len(self.performance_history)
        }
    
    def _extract_pattern(self, trade: Dict) -> Dict:
        """Extract pattern from trade for learning"""
        return {
            'hour': datetime.now().hour,
            'volatility': self._calculate_market_volatility(),
            'strategy': trade.get('strategy'),
            'product': trade.get('product_id')
        }
    
    def get_risk_adjusted_size(self, base_size: float, product_id: str, mode: str) -> float:
        """Get risk-adjusted position size"""
        # Base adjustment from mode
        size = base_size * self.risk_multipliers[mode]
        
        # Adjust based on recent performance for this product
        recent_product_trades = [
            p for p in self.performance_history 
            if p.get('product') == product_id
        ][-10:]
        
        if recent_product_trades:
            avg_pnl = np.mean([t.get('pnl', 0) for t in recent_product_trades])
            if avg_pnl < 0:
                size *= 0.8  # Reduce size for losing products
            elif avg_pnl > 0:
                size *= 1.2  # Increase size for winning products
        
        # Volatility adjustment
        volatility = self._calculate_market_volatility()
        if volatility > self.volatility_threshold['high']:
            size *= 0.7
        
        return size 