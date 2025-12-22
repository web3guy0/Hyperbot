"""
Adaptive Strategy Engine - World-Class Trading Intelligence
Professional-grade strategies used by quantitative hedge funds.

Trend Indicators:
- SupertrendIndicator: Trend-following with dynamic ATR bands
- DonchianChannel: Breakout detection via price channels

Volume Indicators:
- OBVCalculator: On-Balance Volume for accumulation/distribution
- ChaikinMoneyFlow: Money flow based on close position in range

Momentum Indicators:
- StochRSICalculator: Stochastic RSI for sensitive overbought/oversold
"""

from .market_regime import MarketRegimeDetector, MarketRegime
from .smart_money import SmartMoneyAnalyzer, SmartMoneyZone
from .order_flow import OrderFlowAnalyzer
from .multi_timeframe import MultiTimeframeAnalyzer
from .adaptive_risk import AdaptiveRiskManager
from .session_manager import SessionManager, TradingSession
from .multi_asset_correlation import MultiAssetCorrelationAnalyzer, CorrelationState, RelativeStrength
from .supertrend import SupertrendIndicator, SupertrendDirection, SupertrendResult
from .donchian import DonchianChannel, DonchianPosition, DonchianResult
from .stoch_rsi import StochRSICalculator, StochRSIResult
from .obv import OBVCalculator, OBVResult
from .cmf import ChaikinMoneyFlow, CMFResult
from .human_logic import HumanTradingLogic, TrapType, TrapSignal, MeanReversionSignal
from .performance_engine import AdaptivePerformanceEngine, TradeResult, MarketCondition, PerformanceStats

__all__ = [
    # Market Analysis
    'MarketRegimeDetector',
    'MarketRegime',
    'SmartMoneyAnalyzer',
    'SmartMoneyZone',
    'OrderFlowAnalyzer',
    'MultiTimeframeAnalyzer',
    'AdaptiveRiskManager',
    'SessionManager',
    'TradingSession',
    'MultiAssetCorrelationAnalyzer',
    'CorrelationState',
    'RelativeStrength',
    # Trend Indicators
    'SupertrendIndicator',
    'SupertrendDirection',
    'SupertrendResult',
    'DonchianChannel',
    'DonchianPosition',
    'DonchianResult',
    # Momentum Indicators
    'StochRSICalculator',
    'StochRSIResult',
    # Volume Indicators
    'OBVCalculator',
    'OBVResult',
    'ChaikinMoneyFlow',
    'CMFResult',
    # Human-Like Trading Intelligence
    'HumanTradingLogic',
    'TrapType',
    'TrapSignal',
    'MeanReversionSignal',
    # Performance Engine (Self-Learning)
    'AdaptivePerformanceEngine',
    'TradeResult',
    'MarketCondition',
    'PerformanceStats',
]
