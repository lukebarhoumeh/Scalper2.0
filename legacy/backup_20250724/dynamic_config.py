#!/usr/bin/env python3
"""
Dynamic Configuration Manager - HFT Production Grade
Real-time parameter optimization based on market conditions
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import asyncio
import numpy as np
from threading import Lock
import yaml

from market_intelligence import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class TradingParameters:
    """Comprehensive trading parameters with validation"""
    # Engine parameters
    poll_interval_sec: int = 30
    rest_rate_limit: int = 5
    use_ws_feed: bool = True
    
    # Risk parameters
    cooldown_sec: int = 300
    inventory_cap_usd: float = 50.0
    per_coin_position_limit: float = 25.0
    max_position_usd: float = 30.0
    
    # Strategy parameters
    scalper_sma_window: int = 30
    scalper_vol_thresh: float = 15.0
    scalper_spread_thresh: float = 0.8
    breakout_lookback: int = 80
    breakout_atr_window: int = 20
    breakout_atr_mult: float = 2.0
    
    # Strategy weights
    scalper_weight: float = 0.6
    breakout_weight: float = 0.4
    
    # Volatility parameters
    target_vol_pct: float = 10.0
    vol_floor_pct: float = 3.0
    
    # Circuit breakers
    max_drawdown_pct: float = 5.0
    max_daily_loss_usd: float = 100.0
    position_check_interval: int = 60
    
    def validate(self) -> bool:
        """Validate parameter constraints"""
        if self.poll_interval_sec < 10:
            raise ValueError("Poll interval too low - minimum 10 seconds")
        
        if self.inventory_cap_usd < self.per_coin_position_limit:
            raise ValueError("Inventory cap must exceed per-coin limit")
        
        if self.scalper_weight + self.breakout_weight != 1.0:
            raise ValueError("Strategy weights must sum to 1.0")
        
        return True

class DynamicConfigManager:
    """
    Production-grade dynamic configuration system with hot-reloading
    """
    
    def __init__(self, config_path: str = "dynamic_config.yaml"):
        self.config_path = config_path
        self.current_params = TradingParameters()
        self.regime_params = self._load_regime_parameters()
        self.parameter_history = []
        self.lock = Lock()
        
        # Performance tracking
        self.regime_performance = {}
        self.parameter_performance = {}
        
        # Optimization boundaries
        self.param_bounds = {
            'poll_interval_sec': (10, 120),
            'cooldown_sec': (60, 900),
            'inventory_cap_usd': (20, 200),
            'scalper_vol_thresh': (5, 50),
            'scalper_spread_thresh': (0.1, 2.0),
            'breakout_atr_mult': (1.0, 4.0)
        }
        
        logger.info("Dynamic Config Manager initialized")
    
    def _load_regime_parameters(self) -> Dict[str, TradingParameters]:
        """Load optimized parameters for each market regime"""
        return {
            'bull': TradingParameters(
                poll_interval_sec=15,
                cooldown_sec=180,
                inventory_cap_usd=100.0,
                per_coin_position_limit=50.0,
                scalper_vol_thresh=10.0,
                scalper_spread_thresh=0.5,
                breakout_lookback=60,
                breakout_atr_mult=1.8,
                scalper_weight=0.3,
                breakout_weight=0.7,
                vol_floor_pct=2.0
            ),
            'bear': TradingParameters(
                poll_interval_sec=60,
                cooldown_sec=600,
                inventory_cap_usd=40.0,
                per_coin_position_limit=20.0,
                scalper_vol_thresh=30.0,
                scalper_spread_thresh=1.5,
                breakout_lookback=120,
                breakout_atr_mult=3.0,
                scalper_weight=0.8,
                breakout_weight=0.2,
                vol_floor_pct=5.0
            ),
            'volatile': TradingParameters(
                poll_interval_sec=20,
                cooldown_sec=240,
                inventory_cap_usd=60.0,
                per_coin_position_limit=30.0,
                scalper_vol_thresh=20.0,
                scalper_spread_thresh=1.0,
                breakout_lookback=80,
                breakout_atr_mult=2.2,
                scalper_weight=0.5,
                breakout_weight=0.5,
                vol_floor_pct=4.0
            ),
            'sideways': TradingParameters(
                poll_interval_sec=30,
                cooldown_sec=300,
                inventory_cap_usd=75.0,
                per_coin_position_limit=35.0,
                scalper_vol_thresh=15.0,
                scalper_spread_thresh=0.8,
                breakout_lookback=100,
                breakout_atr_mult=2.5,
                scalper_weight=0.7,
                breakout_weight=0.3,
                vol_floor_pct=3.0
            )
        }
    
    def adapt_to_regime(self, regime: MarketRegime) -> TradingParameters:
        """
        Adapt trading parameters based on market regime with intelligent blending
        """
        with self.lock:
            # Get base parameters for regime
            base_params = self.regime_params.get(regime.regime, self.current_params)
            
            # Apply confidence-weighted blending
            if regime.confidence < 0.7:
                # Low confidence - blend with current parameters
                blended_params = self._blend_parameters(
                    base_params, 
                    self.current_params, 
                    regime.confidence
                )
            else:
                blended_params = base_params
            
            # Apply microstructure adjustments
            adjusted_params = self._apply_microstructure_adjustments(
                blended_params, 
                regime
            )
            
            # Apply volatility scaling
            final_params = self._apply_volatility_scaling(
                adjusted_params, 
                regime
            )
            
            # Validate before applying
            final_params.validate()
            
            # Track parameter changes
            self._track_parameter_change(self.current_params, final_params, regime)
            
            # Update current parameters
            self.current_params = final_params
            
            # Save to file for persistence
            self._save_current_config()
            
            logger.info(f"Parameters adapted for {regime.regime} regime (confidence: {regime.confidence:.2f})")
            
            return final_params
    
    def _blend_parameters(self, new_params: TradingParameters, old_params: TradingParameters, weight: float) -> TradingParameters:
        """Intelligent parameter blending with constraints"""
        blended = TradingParameters()
        
        # Blend numeric parameters
        for field in ['poll_interval_sec', 'cooldown_sec', 'inventory_cap_usd', 
                     'scalper_vol_thresh', 'scalper_spread_thresh', 'breakout_atr_mult']:
            old_val = getattr(old_params, field)
            new_val = getattr(new_params, field)
            
            # Weighted average with bounds checking
            blended_val = old_val * (1 - weight) + new_val * weight
            
            # Apply bounds if defined
            if field in self.param_bounds:
                min_val, max_val = self.param_bounds[field]
                blended_val = np.clip(blended_val, min_val, max_val)
            
            setattr(blended, field, type(old_val)(blended_val))
        
        # Strategy weights need special handling to sum to 1
        total_weight = new_params.scalper_weight * weight + old_params.scalper_weight * (1 - weight)
        total_weight += new_params.breakout_weight * weight + old_params.breakout_weight * (1 - weight)
        
        blended.scalper_weight = (new_params.scalper_weight * weight + old_params.scalper_weight * (1 - weight)) / total_weight
        blended.breakout_weight = 1.0 - blended.scalper_weight
        
        return blended
    
    def _apply_microstructure_adjustments(self, params: TradingParameters, regime: MarketRegime) -> TradingParameters:
        """Adjust parameters based on microstructure health"""
        adjusted = TradingParameters(**asdict(params))
        
        # Poor microstructure = more conservative
        if regime.microstructure_score < 0.4:
            adjusted.inventory_cap_usd *= 0.7
            adjusted.per_coin_position_limit *= 0.7
            adjusted.scalper_spread_thresh *= 1.3
            adjusted.cooldown_sec = int(adjusted.cooldown_sec * 1.2)
            logger.info("Applied conservative adjustments due to poor microstructure")
        
        # Excellent microstructure = more aggressive
        elif regime.microstructure_score > 0.8:
            adjusted.inventory_cap_usd *= 1.2
            adjusted.per_coin_position_limit *= 1.2
            adjusted.scalper_spread_thresh *= 0.8
            adjusted.cooldown_sec = int(adjusted.cooldown_sec * 0.8)
            logger.info("Applied aggressive adjustments due to healthy microstructure")
        
        return adjusted
    
    def _apply_volatility_scaling(self, params: TradingParameters, regime: MarketRegime) -> TradingParameters:
        """Scale parameters based on volatility regime"""
        adjusted = TradingParameters(**asdict(params))
        
        vol_multipliers = {
            'low': {'position': 1.3, 'threshold': 0.7, 'cooldown': 0.8},
            'normal': {'position': 1.0, 'threshold': 1.0, 'cooldown': 1.0},
            'high': {'position': 0.7, 'threshold': 1.3, 'cooldown': 1.2},
            'extreme': {'position': 0.4, 'threshold': 2.0, 'cooldown': 1.5}
        }
        
        multipliers = vol_multipliers.get(regime.volatility_regime, vol_multipliers['normal'])
        
        # Apply scaling
        adjusted.inventory_cap_usd *= multipliers['position']
        adjusted.per_coin_position_limit *= multipliers['position']
        adjusted.scalper_vol_thresh *= multipliers['threshold']
        adjusted.cooldown_sec = int(adjusted.cooldown_sec * multipliers['cooldown'])
        
        # Ensure bounds
        adjusted.inventory_cap_usd = np.clip(adjusted.inventory_cap_usd, 20, 200)
        adjusted.per_coin_position_limit = np.clip(adjusted.per_coin_position_limit, 10, 100)
        
        return adjusted
    
    def optimize_parameters(self, performance_data: Dict[str, float]) -> TradingParameters:
        """
        Online parameter optimization using performance feedback
        """
        # Track performance for current parameters
        param_key = self._get_param_key(self.current_params)
        
        if param_key not in self.parameter_performance:
            self.parameter_performance[param_key] = []
        
        self.parameter_performance[param_key].append(performance_data)
        
        # Only optimize if we have enough data
        if len(self.parameter_performance[param_key]) < 10:
            return self.current_params
        
        # Calculate performance metrics
        recent_performance = self.parameter_performance[param_key][-10:]
        avg_pnl = np.mean([p.get('pnl', 0) for p in recent_performance])
        sharpe = self._calculate_sharpe(recent_performance)
        
        # If performance is poor, try parameter perturbation
        if sharpe < 0.5:
            logger.info(f"Poor performance detected (Sharpe: {sharpe:.2f}), optimizing parameters")
            return self._perturb_parameters(self.current_params, sharpe)
        
        return self.current_params
    
    def _perturb_parameters(self, params: TradingParameters, current_sharpe: float) -> TradingParameters:
        """Intelligent parameter perturbation for optimization"""
        perturbed = TradingParameters(**asdict(params))
        
        # Perturbation magnitude based on how bad performance is
        perturbation_scale = max(0.05, min(0.3, 1.0 - current_sharpe))
        
        # Randomly perturb key parameters
        perturb_fields = ['scalper_vol_thresh', 'scalper_spread_thresh', 'cooldown_sec']
        
        for field in perturb_fields:
            if field in self.param_bounds:
                current_val = getattr(params, field)
                min_val, max_val = self.param_bounds[field]
                
                # Random perturbation
                delta = np.random.normal(0, perturbation_scale * (max_val - min_val))
                new_val = np.clip(current_val + delta, min_val, max_val)
                
                setattr(perturbed, field, type(current_val)(new_val))
        
        perturbed.validate()
        return perturbed
    
    def _calculate_sharpe(self, performance_data: List[Dict]) -> float:
        """Calculate Sharpe ratio from performance data"""
        returns = [p.get('return', 0) for p in performance_data]
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 30-second intervals)
        periods_per_year = 365 * 24 * 60 * 2  # 30-second periods
        sharpe = avg_return / std_return * np.sqrt(periods_per_year)
        
        return sharpe
    
    def get_strategy_weights(self, regime: MarketRegime, recent_performance: Optional[Dict] = None) -> Dict[str, float]:
        """
        Dynamic strategy weight allocation based on regime and performance
        """
        base_weights = {
            'scalper': self.current_params.scalper_weight,
            'breakout': self.current_params.breakout_weight
        }
        
        # Adjust based on recent performance if available
        if recent_performance:
            scalper_pnl = recent_performance.get('scalper_pnl', 0)
            breakout_pnl = recent_performance.get('breakout_pnl', 0)
            
            total_pnl = abs(scalper_pnl) + abs(breakout_pnl)
            
            if total_pnl > 0:
                # Performance-weighted adjustment
                scalper_perf = scalper_pnl / total_pnl if scalper_pnl > 0 else 0
                breakout_perf = breakout_pnl / total_pnl if breakout_pnl > 0 else 0
                
                # Blend with base weights
                blend_factor = 0.3  # 30% performance, 70% regime-based
                
                base_weights['scalper'] = (
                    base_weights['scalper'] * (1 - blend_factor) + 
                    scalper_perf * blend_factor
                )
                base_weights['breakout'] = 1.0 - base_weights['scalper']
        
        return base_weights
    
    def should_enable_circuit_breaker(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if circuit breaker should be activated
        """
        reasons = []
        
        # Check drawdown
        if metrics.get('drawdown_pct', 0) > self.current_params.max_drawdown_pct:
            reasons.append(f"Drawdown {metrics['drawdown_pct']:.1f}% exceeds limit")
        
        # Check daily loss
        if metrics.get('daily_loss_usd', 0) > self.current_params.max_daily_loss_usd:
            reasons.append(f"Daily loss ${metrics['daily_loss_usd']:.2f} exceeds limit")
        
        # Check consecutive losses
        if metrics.get('consecutive_losses', 0) > 5:
            reasons.append(f"Consecutive losses: {metrics['consecutive_losses']}")
        
        # Check volatility spike
        if metrics.get('volatility_spike', 0) > 3.0:  # 3 standard deviations
            reasons.append(f"Volatility spike detected: {metrics['volatility_spike']:.1f}σ")
        
        should_trigger = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "All systems normal"
        
        return should_trigger, reason_str
    
    def _track_parameter_change(self, old_params: TradingParameters, new_params: TradingParameters, regime: MarketRegime):
        """Track parameter changes for analysis"""
        change_record = {
            'timestamp': datetime.now(timezone.utc),
            'regime': regime.regime,
            'confidence': regime.confidence,
            'changes': {}
        }
        
        # Record significant changes
        for field in asdict(new_params):
            old_val = getattr(old_params, field)
            new_val = getattr(new_params, field)
            
            if isinstance(old_val, (int, float)):
                if abs(old_val - new_val) / (abs(old_val) + 1e-10) > 0.05:  # 5% change
                    change_record['changes'][field] = {
                        'old': old_val,
                        'new': new_val,
                        'change_pct': (new_val - old_val) / (old_val + 1e-10) * 100
                    }
        
        if change_record['changes']:
            self.parameter_history.append(change_record)
            logger.info(f"Parameter changes: {change_record['changes']}")
    
    def _save_current_config(self):
        """Save current configuration to file"""
        config_dict = asdict(self.current_params)
        config_dict['last_updated'] = datetime.now(timezone.utc).isoformat()
        config_dict['regime_performance'] = self.regime_performance
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def load_config(self) -> TradingParameters:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                # Remove non-parameter fields
                config_dict.pop('last_updated', None)
                config_dict.pop('regime_performance', None)
                
                return TradingParameters(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        
        return TradingParameters()
    
    def _get_param_key(self, params: TradingParameters) -> str:
        """Generate unique key for parameter set"""
        key_fields = ['poll_interval_sec', 'cooldown_sec', 'scalper_vol_thresh', 'scalper_spread_thresh']
        key_values = [str(getattr(params, f)) for f in key_fields]
        return "_".join(key_values)
    
    def get_optimization_report(self) -> Dict:
        """Generate parameter optimization report"""
        report = {
            'current_parameters': asdict(self.current_params),
            'parameter_changes': len(self.parameter_history),
            'last_change': self.parameter_history[-1] if self.parameter_history else None,
            'performance_by_params': {}
        }
        
        # Analyze performance by parameter set
        for param_key, performance_list in self.parameter_performance.items():
            if performance_list:
                avg_pnl = np.mean([p.get('pnl', 0) for p in performance_list])
                sharpe = self._calculate_sharpe(performance_list)
                
                report['performance_by_params'][param_key] = {
                    'trades': len(performance_list),
                    'avg_pnl': avg_pnl,
                    'sharpe_ratio': sharpe
                }
        
        return report 