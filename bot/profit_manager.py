"""Automated Profit Management System

This module handles intelligent profit withdrawal, reinvestment, and capital allocation
to ensure consistent profit extraction while maintaining trading capital.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import logging

from .models import Account


logger = logging.getLogger(__name__)


@dataclass
class ProfitWithdrawal:
    """Record of a profit withdrawal"""
    timestamp: datetime
    amount: float
    balance_before: float
    balance_after: float
    reason: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class CapitalPlan:
    """Capital allocation plan"""
    working_capital: float  # Capital for trading
    reserve_fund: float     # Emergency reserve
    withdrawal: float       # Amount to withdraw
    reinvestment: float     # Amount to reinvest
    reason: str


class ProfitManager:
    """
    Manages profit extraction and capital allocation.
    
    Key features:
    - Automatic profit withdrawal when thresholds are met
    - Dynamic allocation based on performance
    - Maintains minimum working capital
    - Compounds profits during winning streaks
    - Increases withdrawals during losing streaks
    """
    
    def __init__(self, 
                 initial_capital: float,
                 min_working_capital: float = 5000.0,
                 profit_threshold_pct: float = 0.05,  # 5% profit triggers evaluation
                 base_withdrawal_rate: float = 0.40,  # 40% of profits withdrawn
                 max_withdrawal_rate: float = 0.70,   # 70% max withdrawal
                 compound_rate: float = 0.20,        # 20% reinvestment rate
                 withdrawal_cooldown_hours: int = 4): # Hours between withdrawals
        
        self.initial_capital = initial_capital
        self.min_working_capital = min_working_capital
        self.profit_threshold_pct = profit_threshold_pct
        self.base_withdrawal_rate = base_withdrawal_rate
        self.max_withdrawal_rate = max_withdrawal_rate
        self.compound_rate = compound_rate
        self.withdrawal_cooldown_hours = withdrawal_cooldown_hours
        
        # State tracking
        self.total_withdrawn = 0.0
        self.total_reinvested = 0.0
        self.withdrawals: List[ProfitWithdrawal] = []
        
        # Performance tracking for dynamic adjustment
        self.daily_results: List[float] = []  # Daily P&L
        self.winning_days = 0
        self.losing_days = 0
        self.current_streak = 0
        self.peak_balance = initial_capital
        
        # Load history
        self._load_history()
    
    def evaluate_profits(self, account: Account) -> Optional[CapitalPlan]:
        """
        Evaluate current profits and determine if withdrawal is appropriate.
        
        Returns a CapitalPlan if action should be taken, None otherwise.
        """
        current_balance = account.equity
        total_profit = current_balance - self.initial_capital
        profit_rate = total_profit / self.initial_capital
        
        # Update peak tracking
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Check if we meet profit threshold
        if profit_rate < self.profit_threshold_pct:
            return None
        
        # Check cooldown
        if not self._check_cooldown():
            return None
        
        # Calculate dynamic withdrawal rate
        withdrawal_rate = self._calculate_dynamic_rate(account)
        
        # Determine allocations
        withdrawal_amount = total_profit * withdrawal_rate
        
        # Reserve 10% for drawdowns
        reserve_amount = total_profit * 0.10
        
        # Reinvest portion of remaining
        remaining = total_profit - withdrawal_amount - reserve_amount
        reinvest_amount = max(0, remaining * self.compound_rate)
        
        # Ensure minimum working capital
        working_capital_after = current_balance - withdrawal_amount
        if working_capital_after < self.min_working_capital:
            # Reduce withdrawal to maintain minimum
            max_withdrawal = current_balance - self.min_working_capital
            withdrawal_amount = max(0, min(withdrawal_amount, max_withdrawal))
            working_capital_after = current_balance - withdrawal_amount
        
        # Build allocation plan
        reason = self._build_reason(profit_rate, withdrawal_rate)
        
        return CapitalPlan(
            working_capital=working_capital_after,
            reserve_fund=reserve_amount,
            withdrawal=withdrawal_amount,
            reinvestment=reinvest_amount,
            reason=reason
        )
    
    def execute_withdrawal(self, account: Account, plan: CapitalPlan) -> bool:
        """
        Execute a profit withdrawal plan.
        
        Returns True if successful, False otherwise.
        """
        if plan.withdrawal <= 0:
            return False
        
        # Record withdrawal
        withdrawal = ProfitWithdrawal(
            timestamp=datetime.now(timezone.utc),
            amount=plan.withdrawal,
            balance_before=account.equity,
            balance_after=account.equity - plan.withdrawal,
            reason=plan.reason,
            metadata={
                'profit_rate': (account.equity - self.initial_capital) / self.initial_capital,
                'win_rate': self._calculate_win_rate(),
                'current_streak': self.current_streak,
                'reserve_fund': plan.reserve_fund,
                'reinvestment': plan.reinvestment
            }
        )
        
        # Update account (in real implementation, this would transfer to external account)
        account.cash -= plan.withdrawal
        account.equity = account.cash + sum(
            pos.qty * account.last_prices.get(pos.symbol, 0) 
            for pos in account.positions.values()
        )
        
        # Update tracking
        self.total_withdrawn += plan.withdrawal
        self.total_reinvested += plan.reinvestment
        self.withdrawals.append(withdrawal)
        
        # Save state
        self._save_history()
        
        logger.info(f"Profit withdrawal executed: ${plan.withdrawal:.2f} - {plan.reason}")
        
        return True
    
    def update_daily_performance(self, daily_pnl: float) -> None:
        """Update daily performance metrics"""
        self.daily_results.append(daily_pnl)
        
        if daily_pnl > 0:
            self.winning_days += 1
            self.current_streak = max(0, self.current_streak) + 1
        else:
            self.losing_days += 1
            self.current_streak = min(0, self.current_streak) - 1
    
    def get_profit_summary(self) -> Dict:
        """Get comprehensive profit management summary"""
        total_days = self.winning_days + self.losing_days
        win_rate = self._calculate_win_rate()
        
        # Calculate average daily profit
        avg_daily = sum(self.daily_results) / len(self.daily_results) if self.daily_results else 0
        
        # Sharpe approximation
        if len(self.daily_results) > 1:
            returns = self.daily_results
            sharpe = (sum(returns) / len(returns)) / (self._std_dev(returns) + 1e-10) * (252 ** 0.5)
        else:
            sharpe = 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'total_withdrawn': self.total_withdrawn,
            'total_reinvested': self.total_reinvested,
            'withdrawal_count': len(self.withdrawals),
            'last_withdrawal': self.withdrawals[-1].timestamp if self.withdrawals else None,
            'win_rate': win_rate,
            'winning_days': self.winning_days,
            'losing_days': self.losing_days,
            'current_streak': self.current_streak,
            'average_daily_pnl': avg_daily,
            'sharpe_ratio': sharpe,
            'peak_balance': self.peak_balance
        }
    
    def project_future_withdrawals(self, current_balance: float, days: int = 30) -> List[Dict]:
        """Project future withdrawal schedule"""
        projections = []
        
        # Calculate average growth rate
        if len(self.daily_results) > 0:
            avg_daily_return = sum(self.daily_results) / len(self.daily_results) / self.initial_capital
        else:
            avg_daily_return = 0.001  # 0.1% default
        
        # Project balance and withdrawals
        projected_balance = current_balance
        last_withdrawal = self.withdrawals[-1].timestamp if self.withdrawals else datetime.now(timezone.utc)
        
        for day in range(1, days + 1):
            # Project balance growth
            projected_balance *= (1 + avg_daily_return)
            
            # Check if withdrawal would trigger
            profit = projected_balance - self.initial_capital
            profit_rate = profit / self.initial_capital
            
            # Check cooldown from last withdrawal
            next_withdrawal_date = last_withdrawal + timedelta(hours=self.withdrawal_cooldown_hours)
            current_date = datetime.now(timezone.utc) + timedelta(days=day)
            
            if (profit_rate >= self.profit_threshold_pct and 
                current_date >= next_withdrawal_date):
                
                # Estimate withdrawal
                withdrawal_amount = profit * self.base_withdrawal_rate
                
                projections.append({
                    'date': current_date,
                    'projected_balance': projected_balance,
                    'projected_withdrawal': withdrawal_amount,
                    'projected_profit_rate': profit_rate
                })
                
                # Update for next iteration
                projected_balance -= withdrawal_amount
                last_withdrawal = current_date
        
        return projections
    
    def _calculate_dynamic_rate(self, account: Account) -> float:
        """Calculate dynamic withdrawal rate based on performance"""
        
        base_rate = self.base_withdrawal_rate
        win_rate = self._calculate_win_rate()
        
        # Adjust based on win rate
        if win_rate > 0.7:
            # High win rate - can be more aggressive with withdrawals
            rate = base_rate * 1.3
        elif win_rate > 0.6:
            # Good win rate - slightly increase
            rate = base_rate * 1.1
        elif win_rate < 0.4:
            # Poor win rate - reduce withdrawals to preserve capital
            rate = base_rate * 0.7
        else:
            rate = base_rate
        
        # Adjust based on streak
        if self.current_streak > 5:
            # Winning streak - reduce withdrawals to compound
            rate *= 0.8
        elif self.current_streak < -3:
            # Losing streak - increase withdrawals to protect profits
            rate *= 1.2
        
        # Adjust based on drawdown
        drawdown = (self.peak_balance - account.equity) / self.peak_balance
        if drawdown > 0.15:  # 15% drawdown
            # Significant drawdown - preserve capital
            rate *= 0.5
        
        # Cap the rate
        return min(rate, self.max_withdrawal_rate)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        total_days = self.winning_days + self.losing_days
        if total_days == 0:
            return 0.5
        return self.winning_days / total_days
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed"""
        if not self.withdrawals:
            return True
        
        last_withdrawal = self.withdrawals[-1].timestamp
        hours_since = (datetime.now(timezone.utc) - last_withdrawal).total_seconds() / 3600
        
        return hours_since >= self.withdrawal_cooldown_hours
    
    def _build_reason(self, profit_rate: float, withdrawal_rate: float) -> str:
        """Build descriptive reason for withdrawal"""
        
        reasons = []
        
        # Profit level
        if profit_rate > 0.20:
            reasons.append("exceptional profits")
        elif profit_rate > 0.10:
            reasons.append("strong profits")
        else:
            reasons.append("profit threshold met")
        
        # Performance
        win_rate = self._calculate_win_rate()
        if win_rate > 0.7:
            reasons.append("excellent win rate")
        elif win_rate < 0.4:
            reasons.append("capital preservation")
        
        # Streak
        if self.current_streak > 5:
            reasons.append("winning streak")
        elif self.current_streak < -3:
            reasons.append("risk reduction")
        
        # Rate
        if withdrawal_rate > self.base_withdrawal_rate * 1.2:
            reasons.append("increased withdrawal")
        elif withdrawal_rate < self.base_withdrawal_rate * 0.8:
            reasons.append("reduced withdrawal")
        
        return " - ".join(reasons)
    
    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        
        return variance ** 0.5
    
    def _save_history(self) -> None:
        """Save withdrawal history"""
        history_file = Path("logs/profit_history.json")
        history_file.parent.mkdir(exist_ok=True)
        
        data = {
            'initial_capital': self.initial_capital,
            'total_withdrawn': self.total_withdrawn,
            'total_reinvested': self.total_reinvested,
            'winning_days': self.winning_days,
            'losing_days': self.losing_days,
            'current_streak': self.current_streak,
            'peak_balance': self.peak_balance,
            'daily_results': self.daily_results[-100:],  # Keep last 100 days
            'withdrawals': [
                {
                    'timestamp': w.timestamp.isoformat(),
                    'amount': w.amount,
                    'balance_before': w.balance_before,
                    'balance_after': w.balance_after,
                    'reason': w.reason,
                    'metadata': w.metadata
                }
                for w in self.withdrawals[-50:]  # Keep last 50 withdrawals
            ]
        }
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_history(self) -> None:
        """Load withdrawal history"""
        history_file = Path("logs/profit_history.json")
        
        if not history_file.exists():
            return
        
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
            
            self.total_withdrawn = data.get('total_withdrawn', 0.0)
            self.total_reinvested = data.get('total_reinvested', 0.0)
            self.winning_days = data.get('winning_days', 0)
            self.losing_days = data.get('losing_days', 0)
            self.current_streak = data.get('current_streak', 0)
            self.peak_balance = data.get('peak_balance', self.initial_capital)
            self.daily_results = data.get('daily_results', [])
            
            # Load withdrawals
            for w_data in data.get('withdrawals', []):
                withdrawal = ProfitWithdrawal(
                    timestamp=datetime.fromisoformat(w_data['timestamp']),
                    amount=w_data['amount'],
                    balance_before=w_data['balance_before'],
                    balance_after=w_data['balance_after'],
                    reason=w_data['reason'],
                    metadata=w_data.get('metadata', {})
                )
                self.withdrawals.append(withdrawal)
            
            logger.info(f"Loaded profit history: ${self.total_withdrawn:.2f} withdrawn")
            
        except Exception as e:
            logger.error(f"Failed to load profit history: {e}")
