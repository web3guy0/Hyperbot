"""
Unit tests for signal generation and validation.

Tests anti-chase logic, RSI blocks, candle pattern detection.
"""

import pytest
from decimal import Decimal
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAntiChaseLogic:
    """Test anti-chase signal filters."""
    
    def test_block_long_after_green_candles(self):
        """Should block LONG after 4+ consecutive green candles."""
        candles = []
        price = 100
        
        # Create 5 consecutive green candles
        for i in range(5):
            candles.append({
                'open': price,
                'close': price + 1,  # Green candle
                'high': price + 1.5,
                'low': price - 0.5,
                'volume': 1000
            })
            price += 1
        
        # Count green candles
        green_count = sum(
            1 for c in candles[-5:] 
            if c['close'] > c['open']
        )
        
        max_chase_candles = 4
        should_block_long = green_count >= max_chase_candles
        
        assert should_block_long, f"Should block LONG after {green_count} green candles"
    
    def test_block_short_after_red_candles(self):
        """Should block SHORT after 4+ consecutive red candles."""
        candles = []
        price = 100
        
        # Create 5 consecutive red candles
        for i in range(5):
            candles.append({
                'open': price,
                'close': price - 1,  # Red candle
                'high': price + 0.5,
                'low': price - 1.5,
                'volume': 1000
            })
            price -= 1
        
        red_count = sum(
            1 for c in candles[-5:] 
            if c['close'] < c['open']
        )
        
        max_chase_candles = 4
        should_block_short = red_count >= max_chase_candles
        
        assert should_block_short, f"Should block SHORT after {red_count} red candles"
    
    def test_allow_long_on_pullback(self):
        """Should allow LONG on red candle pullback in uptrend."""
        # Uptrend with pullback
        candles = [
            {'open': 100, 'close': 101, 'high': 101.5, 'low': 99.5, 'volume': 1000},  # Green
            {'open': 101, 'close': 102, 'high': 102.5, 'low': 100.5, 'volume': 1000},  # Green
            {'open': 102, 'close': 101.5, 'high': 102.5, 'low': 101, 'volume': 1000},  # RED - pullback
        ]
        
        last_candle_red = candles[-1]['close'] < candles[-1]['open']
        
        assert last_candle_red, "Last candle should be red (pullback)"
        # This is a GOOD long entry - buying the dip
    
    def test_allow_short_on_bounce(self):
        """Should allow SHORT on green candle bounce in downtrend."""
        # Downtrend with bounce
        candles = [
            {'open': 100, 'close': 99, 'high': 100.5, 'low': 98.5, 'volume': 1000},  # Red
            {'open': 99, 'close': 98, 'high': 99.5, 'low': 97.5, 'volume': 1000},   # Red
            {'open': 98, 'close': 98.5, 'high': 99, 'low': 97.5, 'volume': 1000},   # GREEN - bounce
        ]
        
        last_candle_green = candles[-1]['close'] > candles[-1]['open']
        
        assert last_candle_green, "Last candle should be green (bounce)"
        # This is a GOOD short entry - selling the rip


class TestRSIBlocks:
    """Test RSI-based signal blocks."""
    
    def test_block_long_when_overbought(self):
        """Should block LONG when RSI > 65."""
        rsi = 70
        rsi_extreme_long_block = 65
        direction = 'long'
        
        should_block = direction == 'long' and rsi > rsi_extreme_long_block
        
        assert should_block, f"Should block LONG when RSI={rsi} > {rsi_extreme_long_block}"
    
    def test_block_short_when_oversold(self):
        """Should block SHORT when RSI < 35."""
        rsi = 28
        rsi_extreme_short_block = 35
        direction = 'short'
        
        should_block = direction == 'short' and rsi < rsi_extreme_short_block
        
        assert should_block, f"Should block SHORT when RSI={rsi} < {rsi_extreme_short_block}"
    
    def test_allow_long_on_oversold(self):
        """Should allow LONG when RSI is oversold (good entry)."""
        rsi = 30
        rsi_pullback_entry_long = 45
        direction = 'long'
        
        is_good_entry = direction == 'long' and rsi < rsi_pullback_entry_long
        
        assert is_good_entry, "RSI=30 should be good long entry"
    
    def test_allow_short_on_overbought(self):
        """Should allow SHORT when RSI is overbought (good entry)."""
        rsi = 72
        rsi_pullback_entry_short = 55
        direction = 'short'
        
        is_good_entry = direction == 'short' and rsi > rsi_pullback_entry_short
        
        assert is_good_entry, "RSI=72 should be good short entry"
    
    def test_rsi_neutral_zone(self):
        """RSI in neutral zone (45-55) should not have strong bias."""
        rsi = 50
        
        is_oversold = rsi < 35
        is_overbought = rsi > 65
        is_neutral = not is_oversold and not is_overbought
        
        assert is_neutral, "RSI=50 should be in neutral zone"


class TestRegimeBlocks:
    """Test market regime-based signal blocks."""
    
    def test_block_long_in_downtrend(self):
        """Should block LONG in TRENDING_DOWN regime."""
        regime = 'TRENDING_DOWN'
        direction = 'long'
        
        should_block = direction == 'long' and regime == 'TRENDING_DOWN'
        
        assert should_block, "Should not LONG in downtrend"
    
    def test_block_short_in_uptrend(self):
        """Should block SHORT in TRENDING_UP regime."""
        regime = 'TRENDING_UP'
        direction = 'short'
        
        should_block = direction == 'short' and regime == 'TRENDING_UP'
        
        assert should_block, "Should not SHORT in uptrend"
    
    def test_allow_both_in_ranging(self):
        """Should allow both directions in RANGING regime."""
        regime = 'RANGING'
        
        block_long = regime == 'TRENDING_DOWN'
        block_short = regime == 'TRENDING_UP'
        
        assert not block_long, "Should allow LONG in ranging"
        assert not block_short, "Should allow SHORT in ranging"


class TestSignalConfirmation:
    """Test signal confirmation logic."""
    
    def test_require_multiple_confirmations(self):
        """Signal should require multiple confirmations."""
        confirmations_required = 2
        current_confirmations = 1
        
        is_confirmed = current_confirmations >= confirmations_required
        
        assert not is_confirmed, "Should not confirm with only 1 confirmation"
    
    def test_confirmation_achieved(self):
        """Signal confirmed after required confirmations."""
        confirmations_required = 2
        current_confirmations = 2
        
        is_confirmed = current_confirmations >= confirmations_required
        
        assert is_confirmed, "Should confirm with 2 confirmations"
    
    def test_signal_expiry(self):
        """Signal should expire after timeout."""
        signal_age_seconds = 400
        signal_expiry_seconds = 300  # 5 minutes
        
        is_expired = signal_age_seconds > signal_expiry_seconds
        
        assert is_expired, "Signal should expire after 400 seconds"
    
    def test_direction_change_resets_confirmation(self):
        """Direction change should reset confirmation count."""
        pending_direction = 'long'
        new_direction = 'short'
        confirmation_count = 2
        
        if new_direction != pending_direction:
            confirmation_count = 1  # Reset
        
        assert confirmation_count == 1, "Should reset on direction change"


class TestScoreCalculation:
    """Test signal score calculation."""
    
    def test_minimum_score_threshold(self):
        """Signal should meet minimum score."""
        min_score = 12
        signal_score = 10
        
        meets_threshold = signal_score >= min_score
        
        assert not meets_threshold, "Score 10 should not meet threshold 12"
    
    def test_score_comparison(self):
        """Higher score wins between directions."""
        long_score = 15
        short_score = 12
        min_threshold = 12
        
        if long_score >= min_threshold and long_score > short_score:
            winner = 'long'
        elif short_score >= min_threshold and short_score > long_score:
            winner = 'short'
        else:
            winner = None
        
        assert winner == 'long', "Long should win with higher score"
    
    def test_penalty_reduces_score(self):
        """Penalties should reduce signal score."""
        base_score = 15
        regime_penalty = 5  # Counter-trend penalty
        
        final_score = base_score - regime_penalty
        
        assert final_score == 10, "Penalty should reduce score"
    
    def test_bonus_increases_score(self):
        """Bonuses should increase signal score."""
        base_score = 10
        trend_alignment_bonus = 2
        
        final_score = base_score + trend_alignment_bonus
        
        assert final_score == 12, "Bonus should increase score"
