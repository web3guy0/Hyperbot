"""
Adaptive Performance Engine
===========================
Makes the bot LEARN from its results and auto-adjust.

KEY CONCEPT: The bot should get SMARTER over time, not just repeat mistakes.

What this does:
1. Tracks recent performance (last N trades)
2. Auto-adjusts signal threshold based on win rate
3. Identifies which conditions work best
4. Knows when to "sit out" and not trade
5. Adapts to different tokens automatically

This is what separates a profitable bot from a losing one.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Overall market condition assessment."""
    FAVORABLE = "favorable"      # Good conditions, trade normally
    NEUTRAL = "neutral"          # Okay conditions, be selective
    UNFAVORABLE = "unfavorable"  # Bad conditions, reduce size
    DANGEROUS = "dangerous"      # Very bad, stop trading


@dataclass
class TradeResult:
    """Record of a completed trade for learning."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    regime: str
    signal_score: int
    session: str  # asia, london, us, etc.
    timestamp: datetime
    duration_minutes: int
    won: bool = field(init=False)
    
    def __post_init__(self):
        self.won = self.pnl > 0


@dataclass
class PerformanceStats:
    """Performance statistics for a category."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    def update(self, trade: TradeResult):
        """Update stats with new trade."""
        self.total_trades += 1
        self.total_pnl += trade.pnl
        
        if trade.won:
            self.wins += 1
            self.avg_win = (self.avg_win * (self.wins - 1) + trade.pnl) / self.wins
        else:
            self.losses += 1
            self.avg_loss = (self.avg_loss * (self.losses - 1) + abs(trade.pnl)) / self.losses
        
        self.win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0
        
        if self.avg_loss > 0:
            self.profit_factor = (self.avg_win * self.wins) / (self.avg_loss * self.losses) if self.losses > 0 else float('inf')


class AdaptivePerformanceEngine:
    """
    Self-Learning Performance Engine
    
    The bot's "brain" that learns from experience:
    - Tracks what works and what doesn't
    - Auto-adjusts aggressiveness based on recent results
    - Knows when to sit out
    - Adapts to different tokens, sessions, regimes
    
    KEY INSIGHT: A 60% win rate bot that knows when to sit out
    beats an 80% win rate bot that trades in bad conditions.
    """
    
    def __init__(self):
        """Initialize the learning engine."""
        # Configuration
        self.lookback_trades = int(os.getenv('PERF_LOOKBACK_TRADES', '20'))
        self.min_trades_for_adjustment = int(os.getenv('MIN_TRADES_FOR_ADJUST', '5'))
        
        # Win rate thresholds for adjustments
        self.excellent_win_rate = float(os.getenv('EXCELLENT_WIN_RATE', '0.70'))
        self.good_win_rate = float(os.getenv('GOOD_WIN_RATE', '0.55'))
        self.poor_win_rate = float(os.getenv('POOR_WIN_RATE', '0.40'))
        self.terrible_win_rate = float(os.getenv('TERRIBLE_WIN_RATE', '0.25'))
        
        # Trade history
        self.recent_trades: deque = deque(maxlen=100)
        
        # Performance by category
        self.stats_by_symbol: Dict[str, PerformanceStats] = {}
        self.stats_by_regime: Dict[str, PerformanceStats] = {}
        self.stats_by_session: Dict[str, PerformanceStats] = {}
        self.stats_by_direction: Dict[str, PerformanceStats] = {}
        
        # Overall stats
        self.overall_stats = PerformanceStats()
        
        # Current adjustments (applied to strategy)
        self.current_adjustments = {
            'signal_threshold_modifier': 0,  # Add to MIN_SIGNAL_SCORE
            'position_size_modifier': 1.0,   # Multiply position size
            'should_trade': True,            # Master switch
            'condition': MarketCondition.NEUTRAL,
            'reason': 'No data yet',
        }
        
        # Losing streak tracker
        self.consecutive_losses = 0
        self.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5'))
        
        # Today's performance
        self.today_trades = 0
        self.today_pnl = 0.0
        self.today_wins = 0
        self.last_trade_date: Optional[datetime] = None
        
        logger.info("🧠 Adaptive Performance Engine initialized")
        logger.info(f"   Lookback: {self.lookback_trades} trades")
        logger.info(f"   Thresholds: Excellent>{self.excellent_win_rate:.0%}, Good>{self.good_win_rate:.0%}, Poor<{self.poor_win_rate:.0%}")
    
    def record_trade(self, trade: TradeResult):
        """
        Record a completed trade and update all stats.
        
        This is called after every trade closes.
        """
        self.recent_trades.append(trade)
        
        # Update overall stats
        self.overall_stats.update(trade)
        
        # Update category stats
        if trade.symbol not in self.stats_by_symbol:
            self.stats_by_symbol[trade.symbol] = PerformanceStats()
        self.stats_by_symbol[trade.symbol].update(trade)
        
        if trade.regime not in self.stats_by_regime:
            self.stats_by_regime[trade.regime] = PerformanceStats()
        self.stats_by_regime[trade.regime].update(trade)
        
        if trade.session not in self.stats_by_session:
            self.stats_by_session[trade.session] = PerformanceStats()
        self.stats_by_session[trade.session].update(trade)
        
        if trade.direction not in self.stats_by_direction:
            self.stats_by_direction[trade.direction] = PerformanceStats()
        self.stats_by_direction[trade.direction].update(trade)
        
        # Track consecutive losses
        if trade.won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Track today's performance
        today = datetime.now(timezone.utc).date()
        if self.last_trade_date != today:
            # New day - reset daily counters
            self.today_trades = 0
            self.today_pnl = 0.0
            self.today_wins = 0
            self.last_trade_date = today
        
        self.today_trades += 1
        self.today_pnl += trade.pnl
        if trade.won:
            self.today_wins += 1
        
        # Recalculate adjustments
        self._recalculate_adjustments()
        
        logger.info(f"📊 Trade recorded: {trade.symbol} {trade.direction} {'✅' if trade.won else '❌'} ${trade.pnl:+.2f}")
        logger.info(f"   Overall: {self.overall_stats.win_rate:.1%} win rate ({self.overall_stats.total_trades} trades)")
    
    def _recalculate_adjustments(self):
        """
        Recalculate strategy adjustments based on recent performance.
        
        This is the CORE of the adaptive engine.
        """
        recent = list(self.recent_trades)[-self.lookback_trades:]
        
        if len(recent) < self.min_trades_for_adjustment:
            self.current_adjustments['reason'] = f'Need {self.min_trades_for_adjustment} trades, have {len(recent)}'
            return
        
        # Calculate recent stats
        wins = sum(1 for t in recent if t.won)
        recent_win_rate = wins / len(recent)
        recent_pnl = sum(t.pnl for t in recent)
        
        # Determine condition and adjustments
        if self.consecutive_losses >= self.max_consecutive_losses:
            # DANGER: Too many consecutive losses - stop trading
            self.current_adjustments['condition'] = MarketCondition.DANGEROUS
            self.current_adjustments['should_trade'] = False
            self.current_adjustments['signal_threshold_modifier'] = 5  # Much stricter
            self.current_adjustments['position_size_modifier'] = 0.3
            self.current_adjustments['reason'] = f'{self.consecutive_losses} consecutive losses - STOP TRADING'
            logger.warning(f"🛑 DANGER: {self.consecutive_losses} consecutive losses - trading paused")
        
        elif recent_win_rate >= self.excellent_win_rate:
            # EXCELLENT: Keep doing what we're doing
            self.current_adjustments['condition'] = MarketCondition.FAVORABLE
            self.current_adjustments['should_trade'] = True
            self.current_adjustments['signal_threshold_modifier'] = -1  # Slightly more aggressive
            self.current_adjustments['position_size_modifier'] = 1.1
            self.current_adjustments['reason'] = f'Excellent {recent_win_rate:.0%} win rate - slightly aggressive'
        
        elif recent_win_rate >= self.good_win_rate:
            # GOOD: Normal operation
            self.current_adjustments['condition'] = MarketCondition.FAVORABLE
            self.current_adjustments['should_trade'] = True
            self.current_adjustments['signal_threshold_modifier'] = 0
            self.current_adjustments['position_size_modifier'] = 1.0
            self.current_adjustments['reason'] = f'Good {recent_win_rate:.0%} win rate - normal mode'
        
        elif recent_win_rate >= self.poor_win_rate:
            # MEDIOCRE: Be more selective
            self.current_adjustments['condition'] = MarketCondition.NEUTRAL
            self.current_adjustments['should_trade'] = True
            self.current_adjustments['signal_threshold_modifier'] = 2  # Stricter
            self.current_adjustments['position_size_modifier'] = 0.8
            self.current_adjustments['reason'] = f'Mediocre {recent_win_rate:.0%} win rate - being selective'
        
        elif recent_win_rate >= self.terrible_win_rate:
            # POOR: Very selective, reduce risk
            self.current_adjustments['condition'] = MarketCondition.UNFAVORABLE
            self.current_adjustments['should_trade'] = True
            self.current_adjustments['signal_threshold_modifier'] = 4  # Much stricter
            self.current_adjustments['position_size_modifier'] = 0.5
            self.current_adjustments['reason'] = f'Poor {recent_win_rate:.0%} win rate - reducing risk'
            logger.warning(f"⚠️ Poor performance: {recent_win_rate:.0%} win rate - tightening criteria")
        
        else:
            # TERRIBLE: Almost stop trading
            self.current_adjustments['condition'] = MarketCondition.DANGEROUS
            self.current_adjustments['should_trade'] = True  # Still allow but very strict
            self.current_adjustments['signal_threshold_modifier'] = 6  # Very strict
            self.current_adjustments['position_size_modifier'] = 0.3
            self.current_adjustments['reason'] = f'Terrible {recent_win_rate:.0%} win rate - minimal trading'
            logger.error(f"🚨 TERRIBLE: {recent_win_rate:.0%} win rate - consider stopping bot")
    
    def get_adjusted_threshold(self, base_threshold: int) -> int:
        """
        Get the adjusted signal threshold based on performance.
        
        Called by strategy when deciding whether to take a trade.
        """
        modifier = self.current_adjustments['signal_threshold_modifier']
        adjusted = base_threshold + modifier
        
        # Ensure within reasonable bounds
        adjusted = max(6, min(20, adjusted))
        
        if modifier != 0:
            logger.debug(f"   Threshold: {base_threshold} + {modifier} = {adjusted} ({self.current_adjustments['reason']})")
        
        return adjusted
    
    def get_adjusted_position_size(self, base_size: float) -> float:
        """
        Get the adjusted position size based on performance.
        """
        modifier = self.current_adjustments['position_size_modifier']
        adjusted = base_size * modifier
        
        if modifier != 1.0:
            logger.debug(f"   Position size: {base_size:.1f}% × {modifier:.1f} = {adjusted:.1f}%")
        
        return adjusted
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if bot should take trades right now.
        
        Returns (should_trade, reason)
        """
        if not self.current_adjustments['should_trade']:
            return False, self.current_adjustments['reason']
        
        # Check today's performance
        if self.today_trades >= 5 and self.today_wins == 0:
            return False, f"0/{self.today_trades} wins today - taking a break"
        
        if self.today_pnl < -10:  # Lost $10+ today
            return False, f"Down ${abs(self.today_pnl):.2f} today - stopping for the day"
        
        return True, self.current_adjustments['reason']
    
    def get_best_conditions(self) -> Dict[str, Any]:
        """
        Analyze what conditions have been most profitable.
        
        Useful for understanding where the bot performs best.
        """
        results = {
            'best_symbol': None,
            'best_regime': None,
            'best_session': None,
            'best_direction': None,
            'avoid_symbol': None,
            'avoid_regime': None,
            'avoid_session': None,
        }
        
        # Find best/worst by win rate (with minimum trades)
        min_trades = 5
        
        # Best symbol
        symbol_rates = [(s, stats.win_rate, stats.total_trades) 
                       for s, stats in self.stats_by_symbol.items() 
                       if stats.total_trades >= min_trades]
        if symbol_rates:
            symbol_rates.sort(key=lambda x: x[1], reverse=True)
            results['best_symbol'] = symbol_rates[0][0]
            results['avoid_symbol'] = symbol_rates[-1][0] if symbol_rates[-1][1] < 0.4 else None
        
        # Best regime
        regime_rates = [(r, stats.win_rate, stats.total_trades) 
                       for r, stats in self.stats_by_regime.items() 
                       if stats.total_trades >= min_trades]
        if regime_rates:
            regime_rates.sort(key=lambda x: x[1], reverse=True)
            results['best_regime'] = regime_rates[0][0]
            results['avoid_regime'] = regime_rates[-1][0] if regime_rates[-1][1] < 0.4 else None
        
        # Best session
        session_rates = [(s, stats.win_rate, stats.total_trades) 
                        for s, stats in self.stats_by_session.items() 
                        if stats.total_trades >= min_trades]
        if session_rates:
            session_rates.sort(key=lambda x: x[1], reverse=True)
            results['best_session'] = session_rates[0][0]
            results['avoid_session'] = session_rates[-1][0] if session_rates[-1][1] < 0.4 else None
        
        # Best direction
        dir_rates = [(d, stats.win_rate, stats.total_trades) 
                    for d, stats in self.stats_by_direction.items() 
                    if stats.total_trades >= min_trades]
        if dir_rates:
            dir_rates.sort(key=lambda x: x[1], reverse=True)
            results['best_direction'] = dir_rates[0][0]
        
        return results
    
    def should_avoid_condition(
        self,
        symbol: str = None,
        regime: str = None,
        session: str = None,
        direction: str = None,
    ) -> Tuple[bool, str]:
        """
        Check if we should avoid a specific condition based on history.
        
        Called before taking a trade to filter out bad conditions.
        """
        min_trades = 5
        poor_threshold = 0.35
        
        # Check symbol
        if symbol and symbol in self.stats_by_symbol:
            stats = self.stats_by_symbol[symbol]
            if stats.total_trades >= min_trades and stats.win_rate < poor_threshold:
                return True, f"{symbol} has {stats.win_rate:.0%} win rate - avoiding"
        
        # Check regime
        if regime and regime in self.stats_by_regime:
            stats = self.stats_by_regime[regime]
            if stats.total_trades >= min_trades and stats.win_rate < poor_threshold:
                return True, f"Regime {regime} has {stats.win_rate:.0%} win rate - avoiding"
        
        # Check session
        if session and session in self.stats_by_session:
            stats = self.stats_by_session[session]
            if stats.total_trades >= min_trades and stats.win_rate < poor_threshold:
                return True, f"Session {session} has {stats.win_rate:.0%} win rate - avoiding"
        
        return False, ""
    
    def get_status_report(self) -> str:
        """Get a human-readable status report."""
        lines = [
            "📊 PERFORMANCE ENGINE STATUS",
            "=" * 40,
            f"Overall: {self.overall_stats.win_rate:.1%} win rate ({self.overall_stats.total_trades} trades)",
            f"Total P&L: ${self.overall_stats.total_pnl:.2f}",
            f"Today: {self.today_wins}/{self.today_trades} wins, ${self.today_pnl:+.2f}",
            f"Consecutive losses: {self.consecutive_losses}",
            f"",
            f"Current Condition: {self.current_adjustments['condition'].value}",
            f"Threshold modifier: {self.current_adjustments['signal_threshold_modifier']:+d}",
            f"Position modifier: {self.current_adjustments['position_size_modifier']:.1f}x",
            f"Reason: {self.current_adjustments['reason']}",
        ]
        
        # Add best conditions
        best = self.get_best_conditions()
        if best['best_regime']:
            lines.append(f"")
            lines.append(f"Best regime: {best['best_regime']}")
        if best['avoid_regime']:
            lines.append(f"Avoid regime: {best['avoid_regime']}")
        
        return "\n".join(lines)


# Global instance
performance_engine = AdaptivePerformanceEngine()
