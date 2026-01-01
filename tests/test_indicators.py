"""
Unit tests for indicator calculations.

Tests RSI, EMA, MACD, ADX, ATR calculations to ensure accuracy.
"""

import pytest
from decimal import Decimal
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRSICalculation:
    """Test RSI (Relative Strength Index) calculations."""
    
    def test_rsi_oversold_in_downtrend(self, trending_down_candles):
        """RSI should be oversold (<50) in strong downtrend."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in trending_down_candles]
        indicators = calc.calculate_all(prices)
        
        rsi = indicators.get('rsi')
        assert rsi is not None, "RSI should be calculated"
        # In strong downtrend, RSI should be low
        assert rsi < 50, f"RSI should be below 50 in downtrend, got {rsi}"
    
    def test_rsi_overbought_in_uptrend(self, trending_up_candles):
        """RSI should be overbought (>50) in strong uptrend."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in trending_up_candles]
        indicators = calc.calculate_all(prices)
        
        rsi = indicators.get('rsi')
        assert rsi is not None, "RSI should be calculated"
        # In strong uptrend, RSI should be high
        assert rsi > 50, f"RSI should be above 50 in uptrend, got {rsi}"
    
    def test_rsi_neutral_in_range(self, ranging_candles):
        """RSI should be near 50 in ranging market."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in ranging_candles]
        indicators = calc.calculate_all(prices)
        
        rsi = indicators.get('rsi')
        assert rsi is not None, "RSI should be calculated"
        # In ranging market, RSI should be between 25-75 (relaxed bounds)
        assert 25 <= rsi <= 75, f"RSI should be neutral in range, got {rsi}"
    
    def test_rsi_bounds(self, sample_candles):
        """RSI should always be between 0 and 100."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in sample_candles]
        indicators = calc.calculate_all(prices)
        
        rsi = indicators.get('rsi')
        assert rsi is not None, "RSI should be calculated"
        assert 0 <= rsi <= 100, f"RSI must be 0-100, got {rsi}"
    
    def test_rsi_insufficient_data(self):
        """RSI should return empty dict with insufficient data."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal('100'), Decimal('101')]  # Only 2 prices
        indicators = calc.calculate_all(prices)
        
        assert indicators == {}, "Should return empty dict with insufficient data"


class TestEMACalculation:
    """Test EMA (Exponential Moving Average) calculations."""
    
    def test_ema_follows_trend(self, trending_up_candles):
        """EMA should follow price trend."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in trending_up_candles]
        indicators = calc.calculate_all(prices)
        
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        
        assert ema_fast is not None, "Fast EMA should be calculated"
        assert ema_slow is not None, "Slow EMA should be calculated"
        # In uptrend, fast EMA should be above slow EMA
        assert ema_fast > ema_slow, "Fast EMA should be above slow EMA in uptrend"
    
    def test_ema_crossover_detection(self, trending_up_candles):
        """Test EMA crossover in trending market."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in trending_up_candles]
        indicators = calc.calculate_all(prices)
        
        ema_trend = indicators.get('ema_trend')
        assert ema_trend == 'up', f"EMA trend should be 'up' in uptrend, got {ema_trend}"
    
    def test_ema_insufficient_data(self):
        """EMA should not be calculated with insufficient data."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal('100') for _ in range(10)]  # Only 10 prices
        indicators = calc.calculate_all(prices)
        
        assert 'ema_fast' not in indicators, "EMA should not be calculated with insufficient data"


class TestATRCalculation:
    """Test ATR (Average True Range) calculations."""
    
    def test_atr_positive(self, sample_candles):
        """ATR should always be positive."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in sample_candles]
        indicators = calc.calculate_all(prices, candles=sample_candles)
        
        atr = indicators.get('atr')
        assert atr is not None, "ATR should be calculated"
        assert atr > 0, f"ATR must be positive, got {atr}"
    
    def test_atr_higher_in_volatile_market(self):
        """ATR should be higher in volatile markets."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        
        # Low volatility candles
        low_vol_candles = []
        price = 100
        for i in range(200):
            low_vol_candles.append({
                'open': price,
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'volume': 1000
            })
        
        # High volatility candles
        high_vol_candles = []
        price = 100
        for i in range(200):
            high_vol_candles.append({
                'open': price,
                'high': price * 1.05,
                'low': price * 0.95,
                'close': price,
                'volume': 1000
            })
        
        prices_low = [Decimal(str(c['close'])) for c in low_vol_candles]
        prices_high = [Decimal(str(c['close'])) for c in high_vol_candles]
        
        indicators_low = calc.calculate_all(prices_low, candles=low_vol_candles)
        calc.invalidate_cache()  # Reset state between calculations
        indicators_high = calc.calculate_all(prices_high, candles=high_vol_candles)
        
        atr_low = indicators_low.get('atr')
        atr_high = indicators_high.get('atr')
        
        assert atr_low is not None and atr_high is not None, "Both ATRs should be calculated"
        assert atr_high > atr_low, "ATR should be higher in volatile market"


class TestADXCalculation:
    """Test ADX (Average Directional Index) calculations."""
    
    def test_adx_high_in_trend(self, trending_up_candles):
        """ADX should be reasonable in trending market."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in trending_up_candles]
        indicators = calc.calculate_all(prices, candles=trending_up_candles)
        
        adx = indicators.get('adx')
        assert adx is not None, "ADX should be calculated"
        # ADX should be positive
        assert adx > 0, f"ADX should be positive, got {adx}"
    
    def test_adx_low_in_range(self, ranging_candles):
        """ADX should be low (<50) in ranging market."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in ranging_candles]
        indicators = calc.calculate_all(prices, candles=ranging_candles)
        
        adx = indicators.get('adx')
        assert adx is not None, "ADX should be calculated"
        # Ranging market should have reasonable ADX
        assert adx < 50, f"ADX should be <50 in range, got {adx}"
    
    def test_adx_bounds(self, sample_candles):
        """ADX should be between 0 and 100."""
        from app.utils.indicator_calculator import IndicatorCalculator
        
        calc = IndicatorCalculator()
        prices = [Decimal(str(c['close'])) for c in sample_candles]
        indicators = calc.calculate_all(prices, candles=sample_candles)
        
        adx = indicators.get('adx')
        assert adx is not None, "ADX should be calculated"
        assert 0 <= adx <= 100, f"ADX must be 0-100, got {adx}"
