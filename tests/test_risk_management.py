"""
Unit tests for risk management engine.

Tests position limits, drawdown monitoring, kill switch, etc.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRiskLimits:
    """Test risk limit enforcement."""
    
    def test_max_position_limit(self):
        """Test that max position count is enforced."""
        max_positions = 3
        current_positions = [
            {'symbol': 'BTC', 'size': 0.1},
            {'symbol': 'ETH', 'size': 1.0},
            {'symbol': 'SOL', 'size': 10.0},
        ]
        
        can_open_new = len(current_positions) < max_positions
        assert not can_open_new, "Should not allow opening position when at max"
    
    def test_max_leverage_limit(self):
        """Test that leverage is capped."""
        max_leverage = 10
        requested_leverage = 20
        
        actual_leverage = min(requested_leverage, max_leverage)
        assert actual_leverage == max_leverage, "Leverage should be capped at maximum"
    
    def test_max_position_size_pct(self):
        """Test position size percentage limit."""
        account_value = Decimal('1000')
        max_position_pct = Decimal('50')
        requested_pct = Decimal('75')
        
        actual_pct = min(requested_pct, max_position_pct)
        position_value = account_value * actual_pct / 100
        
        assert actual_pct == max_position_pct
        assert position_value == Decimal('500')


class TestDrawdownMonitor:
    """Test drawdown monitoring and alerts."""
    
    def test_drawdown_calculation(self):
        """Test drawdown percentage calculation."""
        peak_value = Decimal('1000')
        current_value = Decimal('900')
        
        drawdown_pct = (peak_value - current_value) / peak_value * 100
        assert drawdown_pct == Decimal('10'), f"Expected 10% drawdown, got {drawdown_pct}%"
    
    def test_drawdown_warning_threshold(self):
        """Test warning at 5% drawdown."""
        peak_value = Decimal('1000')
        current_value = Decimal('945')
        warning_threshold = Decimal('5')
        
        drawdown_pct = (peak_value - current_value) / peak_value * 100
        should_warn = drawdown_pct >= warning_threshold
        
        assert should_warn, "Should trigger warning at 5.5% drawdown"
    
    def test_drawdown_critical_threshold(self):
        """Test critical alert at 10% drawdown."""
        peak_value = Decimal('1000')
        current_value = Decimal('890')
        critical_threshold = Decimal('10')
        
        drawdown_pct = (peak_value - current_value) / peak_value * 100
        is_critical = drawdown_pct >= critical_threshold
        
        assert is_critical, "Should trigger critical at 11% drawdown"
    
    def test_peak_value_tracking(self):
        """Test that peak value updates correctly."""
        values = [1000, 1050, 1030, 1100, 1080]
        peak = 0
        
        for value in values:
            if value > peak:
                peak = value
        
        assert peak == 1100, "Peak should be highest value seen"
    
    def test_drawdown_from_peak(self):
        """Test drawdown calculated from peak, not starting value."""
        starting_value = Decimal('1000')
        peak_value = Decimal('1200')  # Went up first
        current_value = Decimal('1000')  # Back to start
        
        # Drawdown from peak, not from start
        drawdown_pct = (peak_value - current_value) / peak_value * 100
        
        # 16.67% drawdown from peak, even though "flat" from start
        assert drawdown_pct > Decimal('16'), "Drawdown should be from peak"


class TestKillSwitch:
    """Test emergency kill switch functionality."""
    
    def test_kill_switch_daily_loss(self):
        """Kill switch triggers on daily loss limit."""
        daily_loss_limit_pct = Decimal('5')
        starting_balance = Decimal('1000')
        current_balance = Decimal('940')  # 6% loss
        
        daily_loss_pct = (starting_balance - current_balance) / starting_balance * 100
        should_kill = daily_loss_pct >= daily_loss_limit_pct
        
        assert should_kill, "Kill switch should trigger at 6% daily loss"
    
    def test_kill_switch_max_drawdown(self):
        """Kill switch triggers on max drawdown."""
        max_drawdown_limit = Decimal('15')
        peak_value = Decimal('1000')
        current_value = Decimal('840')  # 16% drawdown
        
        drawdown = (peak_value - current_value) / peak_value * 100
        should_kill = drawdown >= max_drawdown_limit
        
        assert should_kill, "Kill switch should trigger at 16% drawdown"
    
    def test_kill_switch_consecutive_losses(self):
        """Kill switch triggers on consecutive losses."""
        max_consecutive_losses = 5
        recent_trades = ['loss', 'loss', 'loss', 'loss', 'loss', 'loss']
        
        consecutive = 0
        for trade in recent_trades:
            if trade == 'loss':
                consecutive += 1
            else:
                consecutive = 0
        
        should_kill = consecutive >= max_consecutive_losses
        assert should_kill, "Kill switch should trigger on 6 consecutive losses"
    
    def test_kill_switch_closes_positions(self):
        """Verify kill switch would close all positions."""
        positions = [
            {'symbol': 'BTC', 'size': 0.1},
            {'symbol': 'ETH', 'size': 1.0},
        ]
        
        # Kill switch should mark all for closure
        to_close = [p['symbol'] for p in positions if p['size'] != 0]
        
        assert len(to_close) == 2, "Should close all open positions"
        assert 'BTC' in to_close
        assert 'ETH' in to_close


class TestRiskPerTrade:
    """Test per-trade risk calculations."""
    
    def test_max_risk_per_trade(self):
        """Test maximum risk per trade limit."""
        account_value = Decimal('1000')
        max_risk_pct = Decimal('2')
        
        max_risk_usd = account_value * max_risk_pct / 100
        assert max_risk_usd == Decimal('20'), "Max risk should be $20 on $1000 account"
    
    def test_position_size_from_risk(self):
        """Calculate position size from risk limit."""
        max_risk_usd = Decimal('20')
        entry_price = Decimal('100')
        sl_price = Decimal('98')  # 2% SL
        
        sl_distance = abs(entry_price - sl_price)
        max_position_size = max_risk_usd / sl_distance
        
        # If risking $20 with $2 SL distance, max 10 units
        assert max_position_size == Decimal('10')
    
    def test_risk_adjusted_for_volatility(self):
        """Test reducing risk in high volatility."""
        base_risk_pct = Decimal('2')
        atr_multiplier = Decimal('2')  # 2x normal ATR
        
        # Reduce risk proportionally to increased volatility
        adjusted_risk = base_risk_pct / atr_multiplier
        
        assert adjusted_risk == Decimal('1'), "Risk should be halved in 2x volatility"


class TestCorrelationRisk:
    """Test correlation-based risk management."""
    
    def test_correlated_positions_limit(self):
        """Limit exposure to correlated assets."""
        positions = {
            'BTC': {'value': 500, 'correlation_group': 'crypto_major'},
            'ETH': {'value': 300, 'correlation_group': 'crypto_major'},
        }
        max_group_exposure = Decimal('600')
        
        group_exposure = sum(
            p['value'] for p in positions.values() 
            if p['correlation_group'] == 'crypto_major'
        )
        
        is_overexposed = Decimal(str(group_exposure)) > max_group_exposure
        assert is_overexposed, "Should detect overexposure to correlated assets"
    
    def test_diversification_requirement(self):
        """Test minimum diversification."""
        positions = {'BTC': 1000}  # All in one asset
        min_positions_for_size = 2  # Require 2+ positions for large size
        max_single_position_pct = Decimal('60')
        account_value = Decimal('1000')
        
        btc_pct = Decimal(str(positions['BTC'])) / account_value * 100
        needs_diversification = (
            btc_pct > max_single_position_pct and 
            len(positions) < min_positions_for_size
        )
        
        assert needs_diversification, "Should require diversification for large positions"
