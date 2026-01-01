"""
Unit tests for PnL (Profit & Loss) calculations.

Critical tests for financial accuracy - these MUST pass.
"""

import pytest
from decimal import Decimal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPnLCalculations:
    """Test PnL percentage calculations."""
    
    def test_long_profit_calculation(self):
        """Long position profit: (exit - entry) / entry * leverage."""
        entry_price = Decimal('100')
        exit_price = Decimal('101')
        leverage = Decimal('10')
        size = Decimal('1')
        
        # Price move: 1%
        price_move_pct = (exit_price - entry_price) / entry_price * 100
        assert price_move_pct == Decimal('1')
        
        # ROE with leverage: 10%
        roe = price_move_pct * leverage
        assert roe == Decimal('10'), f"Expected 10% ROE, got {roe}%"
    
    def test_long_loss_calculation(self):
        """Long position loss calculation."""
        entry_price = Decimal('100')
        exit_price = Decimal('99')
        leverage = Decimal('10')
        
        price_move_pct = (exit_price - entry_price) / entry_price * 100
        roe = price_move_pct * leverage
        
        assert roe == Decimal('-10'), f"Expected -10% ROE, got {roe}%"
    
    def test_short_profit_calculation(self):
        """Short position profit: (entry - exit) / entry * leverage."""
        entry_price = Decimal('100')
        exit_price = Decimal('99')  # Price went down = profit for short
        leverage = Decimal('10')
        
        # For shorts, profit when price goes down
        price_move_pct = (entry_price - exit_price) / entry_price * 100
        assert price_move_pct == Decimal('1')
        
        roe = price_move_pct * leverage
        assert roe == Decimal('10'), f"Expected 10% ROE for short profit, got {roe}%"
    
    def test_short_loss_calculation(self):
        """Short position loss: price went up."""
        entry_price = Decimal('100')
        exit_price = Decimal('101')  # Price went up = loss for short
        leverage = Decimal('10')
        
        price_move_pct = (entry_price - exit_price) / entry_price * 100
        roe = price_move_pct * leverage
        
        assert roe == Decimal('-10'), f"Expected -10% ROE for short loss, got {roe}%"
    
    def test_unrealized_pnl_long(self):
        """Test unrealized PnL calculation for long position."""
        entry_price = Decimal('50000')  # BTC entry
        current_price = Decimal('51000')  # Current price
        size = Decimal('0.1')  # Position size in BTC
        leverage = Decimal('10')
        
        # Unrealized PnL in USD
        unrealized_pnl = (current_price - entry_price) * size
        assert unrealized_pnl == Decimal('100'), f"Expected $100 unrealized PnL, got ${unrealized_pnl}"
        
        # ROE calculation
        position_value = size * entry_price  # $5000
        margin_used = position_value / leverage  # $500
        roe_pct = (unrealized_pnl / margin_used) * 100
        
        assert roe_pct == Decimal('20'), f"Expected 20% ROE, got {roe_pct}%"
    
    def test_unrealized_pnl_short(self):
        """Test unrealized PnL calculation for short position."""
        entry_price = Decimal('50000')
        current_price = Decimal('49000')  # Price dropped = profit for short
        size = Decimal('0.1')
        leverage = Decimal('10')
        
        # Unrealized PnL in USD (short profits when price drops)
        unrealized_pnl = (entry_price - current_price) * size
        assert unrealized_pnl == Decimal('100'), f"Expected $100 unrealized PnL, got ${unrealized_pnl}"
    
    def test_liquidation_price_long(self):
        """Test liquidation price calculation for long."""
        entry_price = Decimal('50000')
        leverage = Decimal('10')
        maintenance_margin = Decimal('0.005')  # 0.5%
        
        # Liquidation occurs when loss = initial margin (simplified)
        # For 10x, initial margin = 10%, so 10% price drop = liquidation
        liq_price = entry_price * (1 - (1 / leverage))
        
        # Expected: 50000 * 0.9 = 45000
        assert liq_price == Decimal('45000'), f"Expected $45000 liq price, got ${liq_price}"
    
    def test_liquidation_price_short(self):
        """Test liquidation price calculation for short."""
        entry_price = Decimal('50000')
        leverage = Decimal('10')
        
        # For shorts, liquidation when price goes UP by margin amount
        liq_price = entry_price * (1 + (1 / leverage))
        
        # Expected: 50000 * 1.1 = 55000
        assert liq_price == Decimal('55000'), f"Expected $55000 liq price, got ${liq_price}"


class TestTPSLCalculations:
    """Test Take Profit / Stop Loss calculations."""
    
    def test_fixed_pnl_tp_long(self):
        """Test fixed PnL TP calculation for long."""
        entry_price = Decimal('100')
        target_pnl_pct = Decimal('10')  # 10% ROE target
        leverage = Decimal('10')
        
        # Price move needed = target_pnl / leverage
        price_move_pct = target_pnl_pct / leverage  # 1%
        tp_price = entry_price * (1 + price_move_pct / 100)
        
        assert tp_price == Decimal('101'), f"Expected TP at $101, got ${tp_price}"
    
    def test_fixed_pnl_sl_long(self):
        """Test fixed PnL SL calculation for long."""
        entry_price = Decimal('100')
        target_sl_pct = Decimal('4')  # 4% max loss
        leverage = Decimal('10')
        
        price_move_pct = target_sl_pct / leverage  # 0.4%
        sl_price = entry_price * (1 - price_move_pct / 100)
        
        expected_sl = Decimal('99.6')
        assert sl_price == expected_sl, f"Expected SL at ${expected_sl}, got ${sl_price}"
    
    def test_fixed_pnl_tp_short(self):
        """Test fixed PnL TP calculation for short."""
        entry_price = Decimal('100')
        target_pnl_pct = Decimal('10')
        leverage = Decimal('10')
        
        price_move_pct = target_pnl_pct / leverage
        # For shorts, TP is BELOW entry
        tp_price = entry_price * (1 - price_move_pct / 100)
        
        assert tp_price == Decimal('99'), f"Expected TP at $99, got ${tp_price}"
    
    def test_fixed_pnl_sl_short(self):
        """Test fixed PnL SL calculation for short."""
        entry_price = Decimal('100')
        target_sl_pct = Decimal('4')
        leverage = Decimal('10')
        
        price_move_pct = target_sl_pct / leverage
        # For shorts, SL is ABOVE entry
        sl_price = entry_price * (1 + price_move_pct / 100)
        
        expected_sl = Decimal('100.4')
        assert sl_price == expected_sl, f"Expected SL at ${expected_sl}, got ${sl_price}"
    
    def test_atr_based_sl(self):
        """Test ATR-based stop loss calculation."""
        entry_price = Decimal('100')
        atr = Decimal('2')  # ATR = $2
        atr_multiplier = Decimal('2')  # 2x ATR
        
        # Long SL
        sl_long = entry_price - (atr * atr_multiplier)
        assert sl_long == Decimal('96'), f"Expected long SL at $96, got ${sl_long}"
        
        # Short SL
        sl_short = entry_price + (atr * atr_multiplier)
        assert sl_short == Decimal('104'), f"Expected short SL at $104, got ${sl_short}"
    
    def test_risk_reward_ratio(self):
        """Test R:R ratio calculation."""
        entry_price = Decimal('100')
        tp_price = Decimal('106')
        sl_price = Decimal('98')
        
        potential_profit = abs(tp_price - entry_price)  # $6
        potential_loss = abs(entry_price - sl_price)    # $2
        
        rr_ratio = potential_profit / potential_loss
        assert rr_ratio == Decimal('3'), f"Expected 3:1 R:R, got {rr_ratio}:1"


class TestPositionSizing:
    """Test position sizing calculations."""
    
    def test_risk_based_position_size(self):
        """Test position size based on risk per trade."""
        account_value = Decimal('1000')
        risk_per_trade_pct = Decimal('2')  # 2% risk
        entry_price = Decimal('100')
        sl_price = Decimal('98')  # 2% from entry
        leverage = Decimal('10')
        
        # Max loss allowed = $20 (2% of $1000)
        max_loss = account_value * risk_per_trade_pct / 100
        
        # SL distance in price
        sl_distance = abs(entry_price - sl_price)
        
        # Position size to risk exactly $20
        position_size_tokens = max_loss / sl_distance
        position_size_usd = position_size_tokens * entry_price
        
        assert position_size_tokens == Decimal('10'), f"Expected 10 tokens, got {position_size_tokens}"
        assert position_size_usd == Decimal('1000'), f"Expected $1000 position, got ${position_size_usd}"
    
    def test_max_position_size_cap(self):
        """Test that position size doesn't exceed maximum."""
        account_value = Decimal('1000')
        max_position_pct = Decimal('50')  # Max 50% of account
        leverage = Decimal('10')
        
        max_position_value = account_value * max_position_pct / 100 * leverage
        
        # With 10x leverage and 50% position, max = $5000
        assert max_position_value == Decimal('5000')
    
    def test_kelly_position_sizing(self):
        """Test Kelly criterion position sizing."""
        win_rate = Decimal('0.6')  # 60% win rate
        avg_win = Decimal('1.5')   # Average win = 1.5x risk
        avg_loss = Decimal('1')    # Average loss = 1x risk
        kelly_fraction = Decimal('0.5')  # Half Kelly
        
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = 1-p, b = win/loss ratio
        b = avg_win / avg_loss
        full_kelly = (win_rate * b - (1 - win_rate)) / b
        
        # Half Kelly for safety
        position_pct = full_kelly * kelly_fraction * 100
        
        # Expected: ((0.6 * 1.5 - 0.4) / 1.5) * 0.5 * 100 = 16.67%
        expected = Decimal('16.666666666666666666666666667')
        assert abs(position_pct - expected) < Decimal('0.01'), f"Kelly sizing: expected ~16.67%, got {position_pct}%"
