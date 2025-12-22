"""
Human-Like Trading Intelligence
===============================
Makes the bot think like a smart human trader who understands:
- Market makers hunt stops before real moves
- Don't chase green candles - wait for pullbacks
- Buy at lows (oversold), sell at highs (overbought)
- Fade the retail crowd's mistakes

KEY PRINCIPLES:
1. DON'T chase momentum (2 green candles = NOT a buy signal)
2. WAIT for liquidity sweeps (stop hunts) then enter opposite
3. USE mean reversion in ranging markets (buy low, sell high)
4. PLACE stops beyond obvious liquidity zones (don't get hunted)

This module overrides momentum-chasing behavior when market is ranging.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TrapType(Enum):
    """Types of retail traps."""
    BULL_TRAP = "bull_trap"  # Fake breakout up, reverses down
    BEAR_TRAP = "bear_trap"  # Fake breakdown, reverses up
    STOP_HUNT_HIGH = "stop_hunt_high"  # Swept highs, reversing
    STOP_HUNT_LOW = "stop_hunt_low"  # Swept lows, reversing


@dataclass
class TrapSignal:
    """Detected trap/sweep signal."""
    trap_type: TrapType
    trigger_price: Decimal
    current_price: Decimal
    swept_level: Decimal
    strength: float  # 0-1
    recommended_direction: str  # 'long' or 'short'
    stop_beyond_level: Decimal  # Safe stop placement
    timestamp: datetime


@dataclass
class MeanReversionSignal:
    """Mean reversion opportunity in ranging market."""
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    target_price: Decimal  # Mean/VWAP
    stop_loss: Decimal
    rsi_value: float
    bb_position: str  # 'lower_band', 'upper_band', 'middle'
    strength: float  # 0-1
    reason: str


class HumanTradingLogic:
    """
    Human-Like Trading Intelligence
    
    Thinks like a professional trader who has been burned by:
    - Chasing green candles
    - Getting stopped out by wicks
    - Fighting the trend
    
    Now trades SMARTER:
    - Fades retail mistakes
    - Enters after liquidity sweeps
    - Uses mean reversion in chop
    - Places stops beyond liquidity
    """
    
    def __init__(self):
        """Initialize human-like trading logic."""
        # Configuration
        self.sweep_lookback = int(os.getenv('SWEEP_LOOKBACK', '20'))
        self.min_sweep_pct = Decimal(os.getenv('MIN_SWEEP_PCT', '0.3'))  # Min 0.3% wick
        self.trap_confirmation_candles = int(os.getenv('TRAP_CONFIRM_CANDLES', '2'))
        
        # Mean reversion thresholds
        self.rsi_oversold = int(os.getenv('RSI_OVERSOLD_EXTREME', '25'))
        self.rsi_overbought = int(os.getenv('RSI_OVERBOUGHT_EXTREME', '75'))
        self.bb_touch_tolerance = Decimal(os.getenv('BB_TOUCH_TOLERANCE', '0.2'))  # % tolerance
        
        # State tracking
        self.recent_traps: deque = deque(maxlen=10)
        self.confirmed_sweeps: deque = deque(maxlen=5)
        self.pending_reversal: Optional[TrapSignal] = None
        self.last_analysis_time: Optional[datetime] = None
        
        # Anti-chase tracking
        self.consecutive_green_candles = 0
        self.consecutive_red_candles = 0
        
        logger.info("🧠 Human Trading Logic initialized")
        logger.info(f"   RSI Extremes: <{self.rsi_oversold} (oversold) | >{self.rsi_overbought} (overbought)")
        logger.info(f"   Min Sweep: {self.min_sweep_pct}%")
    
    def analyze(
        self,
        candles: List[Dict],
        indicators: Dict[str, Any],
        current_price: Decimal,
    ) -> Dict[str, Any]:
        """
        Full human-like analysis.
        
        Returns signals that a smart human would recognize:
        - Trap detection (bull/bear traps)
        - Liquidity sweep signals
        - Mean reversion opportunities
        - Anti-chase warnings
        """
        if len(candles) < 30:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        analysis = {
            'valid': True,
            'timestamp': datetime.now(timezone.utc),
            'traps': [],
            'sweeps': [],
            'mean_reversion': None,
            'anti_chase': None,
            'recommended_action': None,
            'human_bias': None,
        }
        
        # 1. Detect traps (fake breakouts)
        traps = self._detect_traps(candles, current_price)
        analysis['traps'] = traps
        
        # 2. Detect confirmed liquidity sweeps
        sweeps = self._detect_confirmed_sweeps(candles, current_price)
        analysis['sweeps'] = sweeps
        
        # 3. Check for mean reversion opportunity
        mean_rev = self._check_mean_reversion(candles, indicators, current_price)
        analysis['mean_reversion'] = mean_rev
        
        # 4. Anti-chase analysis (don't chase momentum)
        anti_chase = self._anti_chase_analysis(candles, current_price)
        analysis['anti_chase'] = anti_chase
        
        # 5. Determine human-like recommendation
        recommendation = self._get_human_recommendation(
            traps, sweeps, mean_rev, anti_chase, indicators
        )
        analysis['recommended_action'] = recommendation
        analysis['human_bias'] = recommendation.get('bias') if recommendation else None
        
        self.last_analysis_time = datetime.now(timezone.utc)
        return analysis
    
    def _detect_traps(
        self,
        candles: List[Dict],
        current_price: Decimal,
    ) -> List[TrapSignal]:
        """
        Detect bull/bear traps (fake breakouts).
        
        Bull Trap: Price breaks above resistance, then reverses down
        Bear Trap: Price breaks below support, then reverses up
        
        These are GOLD - fade them for easy profits.
        """
        traps = []
        lookback = min(self.sweep_lookback, len(candles) - 3)
        
        if lookback < 5:
            return traps
        
        # Get swing highs and lows from lookback
        lookback_candles = candles[-(lookback + 3):-3]  # Exclude last 3 candles
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in lookback_candles]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in lookback_candles]
        
        if not highs or not lows:
            return traps
        
        resistance = max(highs)
        support = min(lows)
        
        # Check recent candles for trap pattern
        recent = candles[-5:]
        
        for i, candle in enumerate(recent[:-1]):  # Don't check current candle
            c_high = Decimal(str(candle.get('high', candle.get('h', 0))))
            c_low = Decimal(str(candle.get('low', candle.get('l', 0))))
            c_close = Decimal(str(candle.get('close', candle.get('c', 0))))
            c_open = Decimal(str(candle.get('open', candle.get('o', 0))))
            
            # BULL TRAP: Broke above resistance with wick, but closed below
            if c_high > resistance and c_close < resistance:
                # Confirm: Next candle(s) should be bearish
                next_candles = recent[i+1:]
                bearish_confirm = sum(
                    1 for nc in next_candles 
                    if Decimal(str(nc.get('close', nc.get('c', 0)))) < 
                       Decimal(str(nc.get('open', nc.get('o', 0))))
                )
                
                if bearish_confirm >= 1:
                    sweep_pct = (c_high - resistance) / resistance * 100
                    if sweep_pct >= self.min_sweep_pct:
                        trap = TrapSignal(
                            trap_type=TrapType.BULL_TRAP,
                            trigger_price=c_high,
                            current_price=current_price,
                            swept_level=resistance,
                            strength=min(1.0, float(sweep_pct) / 0.5),
                            recommended_direction='short',
                            stop_beyond_level=c_high * Decimal('1.005'),  # 0.5% above wick
                            timestamp=datetime.now(timezone.utc),
                        )
                        traps.append(trap)
                        logger.info(f"🪤 BULL TRAP detected: Swept ${resistance}, reversing down")
            
            # BEAR TRAP: Broke below support with wick, but closed above
            if c_low < support and c_close > support:
                # Confirm: Next candle(s) should be bullish
                next_candles = recent[i+1:]
                bullish_confirm = sum(
                    1 for nc in next_candles 
                    if Decimal(str(nc.get('close', nc.get('c', 0)))) > 
                       Decimal(str(nc.get('open', nc.get('o', 0))))
                )
                
                if bullish_confirm >= 1:
                    sweep_pct = (support - c_low) / support * 100
                    if sweep_pct >= self.min_sweep_pct:
                        trap = TrapSignal(
                            trap_type=TrapType.BEAR_TRAP,
                            trigger_price=c_low,
                            current_price=current_price,
                            swept_level=support,
                            strength=min(1.0, float(sweep_pct) / 0.5),
                            recommended_direction='long',
                            stop_beyond_level=c_low * Decimal('0.995'),  # 0.5% below wick
                            timestamp=datetime.now(timezone.utc),
                        )
                        traps.append(trap)
                        logger.info(f"🪤 BEAR TRAP detected: Swept ${support}, reversing up")
        
        self.recent_traps.extend(traps)
        return traps
    
    def _detect_confirmed_sweeps(
        self,
        candles: List[Dict],
        current_price: Decimal,
    ) -> List[Dict]:
        """
        Detect confirmed liquidity sweeps (stop hunts).
        
        Sweep = Price wicks beyond level, closes back inside
        CONFIRMED = Next candle moves in reversal direction
        
        This is what smart money does: Hunt stops, then reverse.
        """
        sweeps = []
        
        if len(candles) < 10:
            return sweeps
        
        # Get swing points from lookback
        lookback_candles = candles[-self.sweep_lookback:-2]
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in lookback_candles]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in lookback_candles]
        
        if not highs or not lows:
            return sweeps
        
        swing_high = max(highs)
        swing_low = min(lows)
        
        # Check if CURRENT candle is a sweep + confirmation
        prev = candles[-2]
        curr = candles[-1]
        
        prev_high = Decimal(str(prev.get('high', prev.get('h', 0))))
        prev_low = Decimal(str(prev.get('low', prev.get('l', 0))))
        prev_close = Decimal(str(prev.get('close', prev.get('c', 0))))
        
        curr_close = Decimal(str(curr.get('close', curr.get('c', 0))))
        curr_open = Decimal(str(curr.get('open', curr.get('o', 0))))
        
        # BULLISH SWEEP CONFIRMED: Prev swept lows, curr is green
        if prev_low < swing_low and prev_close > swing_low:
            if curr_close > curr_open:  # Current candle is green (confirmation)
                sweep = {
                    'type': 'bullish_sweep_confirmed',
                    'swept_level': float(swing_low),
                    'sweep_low': float(prev_low),
                    'confirmation_close': float(curr_close),
                    'recommended_direction': 'long',
                    'strength': min(1.0, float((swing_low - prev_low) / swing_low * 100)),
                    'safe_stop': float(prev_low * Decimal('0.995')),
                }
                sweeps.append(sweep)
                self.confirmed_sweeps.append(sweep)
                logger.info(f"✅ CONFIRMED BULLISH SWEEP: Stops below ${swing_low} hunted, reversal confirmed")
        
        # BEARISH SWEEP CONFIRMED: Prev swept highs, curr is red
        if prev_high > swing_high and prev_close < swing_high:
            if curr_close < curr_open:  # Current candle is red (confirmation)
                sweep = {
                    'type': 'bearish_sweep_confirmed',
                    'swept_level': float(swing_high),
                    'sweep_high': float(prev_high),
                    'confirmation_close': float(curr_close),
                    'recommended_direction': 'short',
                    'strength': min(1.0, float((prev_high - swing_high) / swing_high * 100)),
                    'safe_stop': float(prev_high * Decimal('1.005')),
                }
                sweeps.append(sweep)
                self.confirmed_sweeps.append(sweep)
                logger.info(f"✅ CONFIRMED BEARISH SWEEP: Stops above ${swing_high} hunted, reversal confirmed")
        
        return sweeps
    
    def _check_mean_reversion(
        self,
        candles: List[Dict],
        indicators: Dict[str, Any],
        current_price: Decimal,
    ) -> Optional[MeanReversionSignal]:
        """
        Check for mean reversion opportunity.
        
        BUY when:
        - RSI < 25 (extremely oversold)
        - Price at/below lower Bollinger Band
        - In ranging market (not trending)
        
        SELL when:
        - RSI > 75 (extremely overbought)
        - Price at/above upper Bollinger Band
        - In ranging market
        
        Target: Return to mean (middle BB or VWAP)
        """
        rsi = indicators.get('rsi')
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        bb_middle = indicators.get('bb_middle')
        vwap = indicators.get('vwap')
        
        if not rsi or not bb_lower or not bb_upper:
            return None
        
        # Convert to Decimal
        bb_upper = Decimal(str(bb_upper))
        bb_lower = Decimal(str(bb_lower))
        bb_middle = Decimal(str(bb_middle)) if bb_middle else (bb_upper + bb_lower) / 2
        
        # Calculate band positions
        band_width = bb_upper - bb_lower
        tolerance = band_width * self.bb_touch_tolerance / 100
        
        # LONG: Oversold + at lower band
        if rsi < self.rsi_oversold and current_price <= (bb_lower + tolerance):
            target = vwap if vwap else float(bb_middle)
            stop = float(bb_lower * Decimal('0.99'))  # 1% below lower band
            
            signal = MeanReversionSignal(
                direction='long',
                entry_price=current_price,
                target_price=Decimal(str(target)),
                stop_loss=Decimal(str(stop)),
                rsi_value=rsi,
                bb_position='lower_band',
                strength=min(1.0, (self.rsi_oversold - rsi) / 10),
                reason=f'RSI={rsi:.0f} oversold + at lower BB - mean reversion LONG',
            )
            logger.info(f"📉 MEAN REVERSION LONG: RSI={rsi:.0f}, Price at lower BB")
            return signal
        
        # SHORT: Overbought + at upper band
        if rsi > self.rsi_overbought and current_price >= (bb_upper - tolerance):
            target = vwap if vwap else float(bb_middle)
            stop = float(bb_upper * Decimal('1.01'))  # 1% above upper band
            
            signal = MeanReversionSignal(
                direction='short',
                entry_price=current_price,
                target_price=Decimal(str(target)),
                stop_loss=Decimal(str(stop)),
                rsi_value=rsi,
                bb_position='upper_band',
                strength=min(1.0, (rsi - self.rsi_overbought) / 10),
                reason=f'RSI={rsi:.0f} overbought + at upper BB - mean reversion SHORT',
            )
            logger.info(f"📈 MEAN REVERSION SHORT: RSI={rsi:.0f}, Price at upper BB")
            return signal
        
        return None
    
    def _anti_chase_analysis(
        self,
        candles: List[Dict],
        current_price: Decimal,
    ) -> Dict[str, Any]:
        """
        Anti-chase analysis - prevent buying after green candles.
        
        PROBLEM: Bot sees 2 green candles → "trend is up!" → LONG
        REALITY: Market makers see retail longs → hunt their stops → dump
        
        SOLUTION: Count consecutive candles, WARN if chasing.
        """
        recent = candles[-10:]
        
        # Count consecutive green/red candles
        green_count = 0
        red_count = 0
        
        for candle in reversed(recent):
            close = Decimal(str(candle.get('close', candle.get('c', 0))))
            open_p = Decimal(str(candle.get('open', candle.get('o', 0))))
            
            if close > open_p:
                if red_count > 0:
                    break
                green_count += 1
            else:
                if green_count > 0:
                    break
                red_count += 1
        
        self.consecutive_green_candles = green_count
        self.consecutive_red_candles = red_count
        
        analysis = {
            'consecutive_green': green_count,
            'consecutive_red': red_count,
            'chasing_long': green_count >= 3,  # Warning: chasing longs
            'chasing_short': red_count >= 3,   # Warning: chasing shorts
            'recommendation': None,
        }
        
        if green_count >= 3:
            analysis['recommendation'] = 'AVOID_LONG'
            analysis['reason'] = f'{green_count} green candles - DO NOT chase longs, wait for pullback'
            logger.warning(f"⚠️ ANTI-CHASE: {green_count} consecutive green candles - DON'T buy here!")
        
        if red_count >= 3:
            analysis['recommendation'] = 'AVOID_SHORT'
            analysis['reason'] = f'{red_count} red candles - DO NOT chase shorts, wait for bounce'
            logger.warning(f"⚠️ ANTI-CHASE: {red_count} consecutive red candles - DON'T short here!")
        
        return analysis
    
    def _get_human_recommendation(
        self,
        traps: List[TrapSignal],
        sweeps: List[Dict],
        mean_rev: Optional[MeanReversionSignal],
        anti_chase: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Get the human-like trading recommendation.
        
        Priority order:
        1. Confirmed sweeps (strongest signal)
        2. Trap signals (fade the fake move)
        3. Mean reversion (buy low/sell high)
        4. Anti-chase warnings (override momentum)
        """
        recommendation = {
            'action': None,
            'direction': None,
            'bias': None,
            'confidence': 0.0,
            'reason': None,
            'stop_placement': None,
        }
        
        # 1. CONFIRMED SWEEPS - Highest priority
        if sweeps:
            latest_sweep = sweeps[-1]
            recommendation['action'] = 'ENTER'
            recommendation['direction'] = latest_sweep['recommended_direction']
            recommendation['bias'] = 'bullish' if latest_sweep['recommended_direction'] == 'long' else 'bearish'
            recommendation['confidence'] = latest_sweep['strength']
            recommendation['reason'] = f"Liquidity sweep confirmed: {latest_sweep['type']}"
            recommendation['stop_placement'] = latest_sweep['safe_stop']
            return recommendation
        
        # 2. TRAP SIGNALS - Second priority
        if traps:
            latest_trap = traps[-1]
            recommendation['action'] = 'ENTER'
            recommendation['direction'] = latest_trap.recommended_direction
            recommendation['bias'] = 'bullish' if latest_trap.recommended_direction == 'long' else 'bearish'
            recommendation['confidence'] = latest_trap.strength
            recommendation['reason'] = f"Trap detected: {latest_trap.trap_type.value}"
            recommendation['stop_placement'] = float(latest_trap.stop_beyond_level)
            return recommendation
        
        # 3. MEAN REVERSION - In ranging markets
        if mean_rev:
            recommendation['action'] = 'ENTER'
            recommendation['direction'] = mean_rev.direction
            recommendation['bias'] = 'bullish' if mean_rev.direction == 'long' else 'bearish'
            recommendation['confidence'] = mean_rev.strength
            recommendation['reason'] = mean_rev.reason
            recommendation['stop_placement'] = float(mean_rev.stop_loss)
            return recommendation
        
        # 4. ANTI-CHASE - Block bad entries
        if anti_chase.get('recommendation'):
            recommendation['action'] = 'WAIT'
            recommendation['direction'] = None
            recommendation['bias'] = None
            recommendation['confidence'] = 0.8
            recommendation['reason'] = anti_chase['reason']
            return recommendation
        
        return None
    
    def should_override_momentum_signal(
        self,
        direction: str,
        candles: List[Dict],
        indicators: Dict[str, Any],
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Check if a momentum signal should be overridden by human logic.
        
        Call this BEFORE taking a momentum-based trade.
        
        Returns:
            Tuple of (should_override, reason, alternative_signal)
        """
        current_price = Decimal(str(candles[-1].get('close', candles[-1].get('c', 0))))
        
        # Run full analysis
        analysis = self.analyze(candles, indicators, current_price)
        
        # Check anti-chase
        anti_chase = analysis.get('anti_chase', {})
        
        if direction == 'long' and anti_chase.get('chasing_long'):
            return True, anti_chase['reason'], None
        
        if direction == 'short' and anti_chase.get('chasing_short'):
            return True, anti_chase['reason'], None
        
        # Check if human logic suggests opposite direction
        recommendation = analysis.get('recommended_action')
        if recommendation and recommendation.get('direction'):
            rec_dir = recommendation['direction']
            if rec_dir != direction and recommendation['confidence'] > 0.6:
                return True, f"Human logic suggests {rec_dir}: {recommendation['reason']}", recommendation
        
        return False, "", None
    
    def get_smart_stop_placement(
        self,
        direction: str,
        entry_price: Decimal,
        candles: List[Dict],
        default_sl_pct: float,
    ) -> Decimal:
        """
        Get smart stop placement beyond liquidity zones.
        
        Instead of obvious support/resistance, place stops BEYOND
        where stop hunters will sweep.
        
        Args:
            direction: 'long' or 'short'
            entry_price: Entry price
            candles: Recent candles
            default_sl_pct: Default stop loss percentage
            
        Returns:
            Smart stop loss price
        """
        if len(candles) < 10:
            # Fallback to default
            if direction == 'long':
                return entry_price * (1 - Decimal(str(default_sl_pct / 100)))
            else:
                return entry_price * (1 + Decimal(str(default_sl_pct / 100)))
        
        recent = candles[-20:]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in recent]
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in recent]
        
        if direction == 'long':
            # Find swing low - place stop BELOW it (beyond liquidity)
            swing_low = min(lows)
            # Add buffer beyond the obvious stop zone (0.5-1%)
            buffer = swing_low * Decimal('0.007')  # 0.7% beyond swing low
            smart_stop = swing_low - buffer
            
            # But don't go too far - cap at 2x default
            max_stop = entry_price * (1 - Decimal(str(default_sl_pct * 2 / 100)))
            return max(smart_stop, max_stop)
        
        else:  # short
            # Find swing high - place stop ABOVE it
            swing_high = max(highs)
            buffer = swing_high * Decimal('0.007')  # 0.7% beyond swing high
            smart_stop = swing_high + buffer
            
            # Cap at 2x default
            max_stop = entry_price * (1 + Decimal(str(default_sl_pct * 2 / 100)))
            return min(smart_stop, max_stop)


# Global instance for easy access
human_logic = HumanTradingLogic()
