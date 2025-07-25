#!/usr/bin/env python3
"""
Advanced Profit Management System
=================================
Implements intelligent profit withdrawal and reinvestment strategies
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ProfitRecord:
    """Individual profit record"""
    timestamp: datetime
    amount: float
    source: str  # 'trading', 'withdrawal', 'reinvestment'
    balance_after: float
    metadata: Dict = field(default_factory=dict)


@dataclass 
class CapitalAllocation:
    """Capital allocation recommendation"""
    working_capital: float
    reserve_capital: float
    withdrawal_amount: float
    reinvestment_amount: float
    reasoning: str


class AdvancedProfitManager:
    """
    Sophisticated profit management with dynamic withdrawal and reinvestment
    Based on performance metrics and market conditions
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.working_capital = initial_capital
        self.reserve_capital = 0.0
        
        # Profit tracking
        self.total_profit = 0.0
        self.withdrawn_profit = 0.0
        self.reinvested_profit = 0.0
        self.profit_history: List[ProfitRecord] = []
        
        # Performance metrics
        self.winning_days = 0
        self.losing_days = 0
        self.current_streak = 0
        self.max_drawdown = 0.0
        self.peak_balance = initial_capital
        
        # Configuration
        self.config = {
            'min_profit_threshold': 0.10,      # 10% profit to trigger withdrawal
            'base_withdrawal_rate': 0.25,      # 25% base withdrawal
            'max_withdrawal_rate': 0.75,       # 75% max withdrawal
            'reinvestment_threshold': 0.20,    # 20% profit for reinvestment
            'compound_rate': 0.15,             # 15% compound back into trading
            'reserve_target': 0.25,            # 25% of profits as reserve
            'min_working_capital': 500.0,      # Minimum trading capital
            'withdrawal_cooldown_hours': 24    # Hours between withdrawals
        }
        
        # Load historical data if exists
        self._load_history()
        
    def update_balance(self, new_balance: float, daily_pnl: float):
        """Update balance and check for profit actions"""
        self.current_balance = new_balance
        
        # Update peak and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update daily stats
        if daily_pnl > 0:
            self.winning_days += 1
            self.current_streak = max(0, self.current_streak) + 1
        else:
            self.losing_days += 1
            self.current_streak = min(0, self.current_streak) - 1
            
        # Check profit thresholds
        total_profit = new_balance - self.initial_capital
        profit_rate = total_profit / self.initial_capital
        
        if profit_rate >= self.config['min_profit_threshold']:
            allocation = self._calculate_allocation(total_profit, daily_pnl)
            self._execute_allocation(allocation)
            
    def _calculate_allocation(self, total_profit: float, recent_pnl: float) -> CapitalAllocation:
        """Calculate optimal capital allocation"""
        # Base withdrawal rate
        withdrawal_rate = self.config['base_withdrawal_rate']
        
        # Adjust based on win rate
        win_rate = self.winning_days / max(1, self.winning_days + self.losing_days)
        
        if win_rate > 0.7:  # High win rate
            withdrawal_rate = self.config['max_withdrawal_rate']
            reasoning = f"High win rate ({win_rate:.1%}) - maximizing withdrawal"
        elif win_rate > 0.6:
            withdrawal_rate = 0.50
            reasoning = f"Good win rate ({win_rate:.1%}) - moderate withdrawal"
        elif win_rate < 0.4:
            withdrawal_rate = 0.15
            reasoning = f"Low win rate ({win_rate:.1%}) - minimal withdrawal"
        else:
            reasoning = f"Average win rate ({win_rate:.1%}) - standard withdrawal"
            
        # Adjust based on streak
        if self.current_streak > 5:
            withdrawal_rate *= 0.8  # Keep more capital during winning streak
            reasoning += ", reducing withdrawal due to winning streak"
        elif self.current_streak < -3:
            withdrawal_rate *= 1.2  # Withdraw more during losing streak
            reasoning += ", increasing withdrawal due to losing streak"
            
        # Calculate amounts
        withdrawal_amount = total_profit * withdrawal_rate
        
        # Reserve allocation
        reserve_amount = total_profit * self.config['reserve_target']
        
        # Reinvestment (remaining after withdrawal and reserve)
        remaining = total_profit - withdrawal_amount - reserve_amount
        reinvestment_amount = max(0, remaining * self.config['compound_rate'])
        
        # Ensure minimum working capital
        working_capital = self.initial_capital + reinvestment_amount
        if working_capital < self.config['min_working_capital']:
            # Reduce withdrawal to maintain minimum
            adjustment = self.config['min_working_capital'] - working_capital
            withdrawal_amount = max(0, withdrawal_amount - adjustment)
            working_capital = self.config['min_working_capital']
            
        return CapitalAllocation(
            working_capital=working_capital,
            reserve_capital=reserve_amount,
            withdrawal_amount=withdrawal_amount,
            reinvestment_amount=reinvestment_amount,
            reasoning=reasoning
        )
        
    def _execute_allocation(self, allocation: CapitalAllocation):
        """Execute the capital allocation"""
        # Check cooldown
        if self.profit_history:
            last_withdrawal = next(
                (p for p in reversed(self.profit_history) if p.source == 'withdrawal'),
                None
            )
            if last_withdrawal:
                hours_since = (datetime.now() - last_withdrawal.timestamp).total_seconds() / 3600
                if hours_since < self.config['withdrawal_cooldown_hours']:
                    logger.info(f"Withdrawal cooldown active ({hours_since:.1f}h)")
                    return
                    
        # Execute withdrawal
        if allocation.withdrawal_amount > 0:
            self.withdrawn_profit += allocation.withdrawal_amount
            self.working_capital -= allocation.withdrawal_amount
            
            record = ProfitRecord(
                timestamp=datetime.now(),
                amount=allocation.withdrawal_amount,
                source='withdrawal',
                balance_after=self.working_capital,
                metadata={
                    'reasoning': allocation.reasoning,
                    'win_rate': self.winning_days / max(1, self.winning_days + self.losing_days)
                }
            )
            self.profit_history.append(record)
            
            logger.info(f"Profit withdrawal: ${allocation.withdrawal_amount:.2f} - {allocation.reasoning}")
            
        # Update reserves
        self.reserve_capital = allocation.reserve_capital
        
        # Track reinvestment
        if allocation.reinvestment_amount > 0:
            self.reinvested_profit += allocation.reinvestment_amount
            
            record = ProfitRecord(
                timestamp=datetime.now(),
                amount=allocation.reinvestment_amount,
                source='reinvestment',
                balance_after=self.working_capital,
                metadata={'type': 'compound'}
            )
            self.profit_history.append(record)
            
        # Save state
        self._save_history()
        
    def get_withdrawal_schedule(self) -> List[Dict]:
        """Get historical and projected withdrawals"""
        schedule = []
        
        # Historical withdrawals
        for record in self.profit_history:
            if record.source == 'withdrawal':
                schedule.append({
                    'date': record.timestamp,
                    'amount': record.amount,
                    'type': 'historical',
                    'reasoning': record.metadata.get('reasoning', '')
                })
                
        # Project next withdrawal
        if self.current_balance > self.initial_capital * 1.1:
            profit = self.current_balance - self.initial_capital
            next_withdrawal = profit * self.config['base_withdrawal_rate']
            
            schedule.append({
                'date': datetime.now() + timedelta(days=1),
                'amount': next_withdrawal,
                'type': 'projected',
                'reasoning': 'Based on current profit level'
            })
            
        return schedule
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        total_days = self.winning_days + self.losing_days
        win_rate = self.winning_days / max(1, total_days)
        
        # Calculate Sharpe ratio approximation
        if self.profit_history:
            daily_returns = []
            for i in range(1, len(self.profit_history)):
                if self.profit_history[i].source == 'trading':
                    prev_balance = self.profit_history[i-1].balance_after
                    curr_balance = self.profit_history[i].balance_after
                    daily_return = (curr_balance - prev_balance) / prev_balance
                    daily_returns.append(daily_return)
                    
            if daily_returns:
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
            
        return {
            'initial_capital': self.initial_capital,
            'current_balance': self.current_balance,
            'working_capital': self.working_capital,
            'reserve_capital': self.reserve_capital,
            'total_profit': self.current_balance - self.initial_capital,
            'profit_rate': (self.current_balance - self.initial_capital) / self.initial_capital,
            'withdrawn_profit': self.withdrawn_profit,
            'reinvested_profit': self.reinvested_profit,
            'win_rate': win_rate,
            'winning_days': self.winning_days,
            'losing_days': self.losing_days,
            'current_streak': self.current_streak,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe,
            'days_active': total_days
        }
        
    def should_increase_risk(self) -> Tuple[bool, float]:
        """Determine if risk should be increased based on performance"""
        summary = self.get_performance_summary()
        
        # Default: no change
        should_increase = False
        multiplier = 1.0
        
        # Strong performance indicators
        if (summary['win_rate'] > 0.65 and 
            summary['sharpe_ratio'] > 1.5 and
            summary['max_drawdown'] < 0.10 and
            self.current_streak > 3):
            
            should_increase = True
            multiplier = 1.25
            
        # Moderate performance
        elif (summary['win_rate'] > 0.55 and
              summary['sharpe_ratio'] > 1.0 and
              summary['max_drawdown'] < 0.15):
            
            should_increase = True
            multiplier = 1.10
            
        # Poor performance - reduce risk
        elif (summary['win_rate'] < 0.45 or
              summary['sharpe_ratio'] < 0.5 or
              summary['max_drawdown'] > 0.20):
            
            should_increase = False
            multiplier = 0.80
            
        return should_increase, multiplier
        
    def get_capital_projection(self, days: int = 30) -> Dict:
        """Project capital growth"""
        summary = self.get_performance_summary()
        
        # Calculate average daily return
        if summary['days_active'] > 0:
            total_return = summary['profit_rate']
            daily_return = (1 + total_return) ** (1 / summary['days_active']) - 1
        else:
            daily_return = 0.0
            
        # Risk-adjusted projection
        risk_adjusted_return = daily_return * summary['win_rate']
        
        projections = {}
        current = self.current_balance
        
        for day in [7, 14, 30, 60, 90]:
            if day <= days:
                projected = current * (1 + risk_adjusted_return) ** day
                projections[f"{day}_days"] = {
                    'balance': projected,
                    'profit': projected - self.initial_capital,
                    'return': (projected - self.initial_capital) / self.initial_capital
                }
                
        return projections
        
    def _save_history(self):
        """Save profit history to file"""
        history_file = Path('data/profit_history.json')
        history_file.parent.mkdir(exist_ok=True)
        
        data = {
            'initial_capital': self.initial_capital,
            'current_balance': self.current_balance,
            'withdrawn_profit': self.withdrawn_profit,
            'reinvested_profit': self.reinvested_profit,
            'winning_days': self.winning_days,
            'losing_days': self.losing_days,
            'history': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'amount': r.amount,
                    'source': r.source,
                    'balance_after': r.balance_after,
                    'metadata': r.metadata
                }
                for r in self.profit_history
            ]
        }
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _load_history(self):
        """Load profit history from file"""
        history_file = Path('data/profit_history.json')
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    
                self.withdrawn_profit = data.get('withdrawn_profit', 0.0)
                self.reinvested_profit = data.get('reinvested_profit', 0.0)
                self.winning_days = data.get('winning_days', 0)
                self.losing_days = data.get('losing_days', 0)
                
                for record_data in data.get('history', []):
                    record = ProfitRecord(
                        timestamp=datetime.fromisoformat(record_data['timestamp']),
                        amount=record_data['amount'],
                        source=record_data['source'],
                        balance_after=record_data['balance_after'],
                        metadata=record_data.get('metadata', {})
                    )
                    self.profit_history.append(record)
                    
                logger.info(f"Loaded {len(self.profit_history)} profit records")
                
            except Exception as e:
                logger.error(f"Failed to load profit history: {e}")
                

# Global instance
_profit_manager = None

def get_profit_manager(initial_capital: float = 1000.0) -> AdvancedProfitManager:
    """Get or create global profit manager"""
    global _profit_manager
    if _profit_manager is None:
        _profit_manager = AdvancedProfitManager(initial_capital)
    return _profit_manager 