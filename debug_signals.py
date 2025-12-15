#!/usr/bin/env python3
"""
Signal Generation Deep Debug
Tests why signals are not generating
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datetime import datetime

# Colors
class C:
    G = '\033[92m'  # Green
    R = '\033[91m'  # Red
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    M = '\033[95m'  # Magenta
    C = '\033[96m'  # Cyan
    E = '\033[0m'   # End

def ok(msg): print(f"{C.G}✅ {msg}{C.E}")
def err(msg): print(f"{C.R}❌ {msg}{C.E}")
def warn(msg): print(f"{C.Y}⚠️  {msg}{C.E}")
def info(msg): print(f"{C.B}ℹ️  {msg}{C.E}")
def hdr(msg): print(f"\n{C.C}{'='*60}\n  {msg}\n{'='*60}{C.E}\n")

async def main():
    hdr("SIGNAL GENERATION DEEP DEBUG")
    
    # Initialize client
    from app.hl.hl_client import HyperLiquidClient
    
    account = os.getenv('ACCOUNT_ADDRESS')
    secret = os.getenv('API_SECRET')
    
    client = HyperLiquidClient(account, secret, secret, testnet=False)
    ok("HyperLiquid client connected")
    
    # Test with BTC
    symbol = "BTC"
    hdr(f"Testing {symbol} Signal Generation")
    
    # Fetch candles
    candles = client.get_candles(symbol, "15m", 200)
    if not candles or len(candles) < 100:
        err(f"Not enough candles: {len(candles) if candles else 0}")
        return
    
    ok(f"Fetched {len(candles)} candles")
    
    # Create DataFrame
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    # Already has open, high, low, close, volume
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    latest_price = df['close'].iloc[-1]
    info(f"Latest price: ${latest_price:,.2f}")
    info(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Initialize strategy
    from app.strategies.rule_based.swing_strategy import SwingStrategy
    
    strategy = SwingStrategy(symbol=symbol)
    ok("SwingStrategy initialized")
    
    # Show config
    hdr("Strategy Configuration")
    print(f"  Min Signal Score: {strategy.min_signal_score}/{strategy.max_signal_score} ({strategy.min_signal_score/strategy.max_signal_score:.0%})")
    print(f"  Confirmation Scans: {os.getenv('SIGNAL_CONFIRMATION_SCANS', 3)}")
    print(f"  Direction Lock: {os.getenv('DIRECTION_LOCK_SECONDS', 900)}s")
    
    # Test regime detection
    hdr("Market Regime Detection")
    try:
        regime = strategy.market_regime_detector.detect(df)
        print(f"  Regime: {C.M}{regime.value}{C.E}")
        
        # Show regime details
        detector = strategy.market_regime_detector
        
        # Calculate indicators
        close = df['close']
        high = df['high']
        low = df['low']
        
        # EMAs
        ema_fast = close.ewm(span=9).mean().iloc[-1]
        ema_slow = close.ewm(span=21).mean().iloc[-1]
        ema_diff_pct = (ema_fast - ema_slow) / ema_slow * 100
        
        # ADX (simple approximation)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        print(f"  EMA9: ${ema_fast:,.2f}")
        print(f"  EMA21: ${ema_slow:,.2f}")
        print(f"  EMA Diff: {ema_diff_pct:+.2f}%")
        print(f"  ATR14: ${atr:,.2f} ({atr/latest_price*100:.2f}%)")
        
    except Exception as e:
        err(f"Regime detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test scoring
    hdr("Signal Scoring")
    try:
        long_score, short_score = strategy._calculate_scores(df)
        
        threshold = strategy.min_signal_score
        max_score = strategy.max_signal_score
        
        long_pct = long_score / max_score * 100
        short_pct = short_score / max_score * 100
        threshold_pct = threshold / max_score * 100
        
        # Visual bar
        def score_bar(score, max_s=max_score):
            filled = int(score / max_s * 20)
            return '█' * filled + '░' * (20 - filled)
        
        print(f"  {C.G}LONG:  [{score_bar(long_score)}] {long_score}/{max_score} ({long_pct:.0f}%){C.E}")
        print(f"  {C.R}SHORT: [{score_bar(short_score)}] {short_score}/{max_score} ({short_pct:.0f}%){C.E}")
        print(f"  {C.Y}THRESHOLD: {threshold}/{max_score} ({threshold_pct:.0f}%){C.E}")
        
        if long_score >= threshold:
            ok(f"LONG score meets threshold!")
        elif short_score >= threshold:
            ok(f"SHORT score meets threshold!")
        else:
            warn(f"Both scores below threshold - NO SIGNAL")
            gap_long = threshold - long_score
            gap_short = threshold - short_score
            info(f"  Gap to threshold: LONG needs +{gap_long}, SHORT needs +{gap_short}")
        
    except Exception as e:
        err(f"Score calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test individual components
    hdr("Individual Component Scores")
    try:
        # Get detailed breakdown
        components = {
            'RSI': None,
            'EMA Trend': None,
            'MACD': None,
            'Bollinger': None,
            'Volume': None,
            'Regime': None,
        }
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        
        print(f"  RSI(14): {rsi_val:.1f}")
        if rsi_val < 30:
            print(f"    {C.G}→ Oversold (bullish){C.E}")
        elif rsi_val > 70:
            print(f"    {C.R}→ Overbought (bearish){C.E}")
        else:
            print(f"    → Neutral")
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        print(f"\n  MACD: {macd.iloc[-1]:.2f}")
        print(f"  Signal: {signal.iloc[-1]:.2f}")
        print(f"  Histogram: {histogram.iloc[-1]:.2f}")
        if histogram.iloc[-1] > 0:
            print(f"    {C.G}→ Bullish (MACD above signal){C.E}")
        else:
            print(f"    {C.R}→ Bearish (MACD below signal){C.E}")
        
        # Supertrend
        print(f"\n  Checking Supertrend...")
        try:
            supertrend_result = strategy.supertrend.calculate(df)
            st_direction = supertrend_result.get('direction', 'unknown')
            st_strength = supertrend_result.get('strength', 0)
            print(f"  Supertrend Direction: {st_direction}")
            print(f"  Supertrend Strength: {st_strength:.2f}")
        except Exception as e:
            warn(f"  Supertrend error: {e}")
        
    except Exception as e:
        err(f"Component analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test signal generation
    hdr("Full Signal Generation Test")
    try:
        signal = await strategy.generate_signal(symbol, df)
        
        if signal:
            ok(f"Signal Generated!")
            print(f"  Direction: {signal.get('direction')}")
            print(f"  Confidence: {signal.get('confidence', 0):.1%}")
            print(f"  Score: {signal.get('score', 0)}")
            print(f"  Entry: ${signal.get('entry', 0):,.2f}")
            print(f"  Stop Loss: ${signal.get('stop_loss', 0):,.2f}")
            print(f"  Take Profit: ${signal.get('take_profit', 0):,.2f}")
        else:
            warn("No signal generated")
            info("Possible reasons:")
            info("  1. Scores below threshold")
            info("  2. Regime blocking (counter-trend)")
            info("  3. Direction lock active")
            info("  4. Confirmation not met")
            
    except Exception as e:
        err(f"Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Recommendations
    hdr("Recommendations")
    
    if long_score < threshold and short_score < threshold:
        print(f"  {C.Y}Current threshold ({threshold}/{max_score}) may be too high{C.E}")
        print(f"  Consider lowering to {max(long_score, short_score) - 1} for testing")
        print(f"\n  To lower threshold, set in .env:")
        print(f"  {C.C}MIN_SIGNAL_SCORE={max(long_score, short_score) - 1}{C.E}")
    
    print(f"\n  Current time: {datetime.now()}")
    print(f"  Market may be in consolidation - wait for trending conditions")
    
    hdr("Done")

if __name__ == "__main__":
    asyncio.run(main())
