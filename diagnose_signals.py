#!/usr/bin/env python3
"""
Signal Diagnostic Tool
Shows exactly WHY signals are being blocked or scored low.
"""
import asyncio
import os
from decimal import Decimal
from datetime import datetime, timezone

# Load .env FIRST
from dotenv import load_dotenv
load_dotenv()

# Set up before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')

from app.hl.hl_client import HyperLiquidClient
from app.strategies.rule_based.swing_strategy import SwingStrategy
from app.strategies.adaptive.market_regime import MarketRegime


async def diagnose_signals():
    """Run full signal diagnosis"""
    
    symbol = os.getenv('SYMBOL', 'SOL')
    print(f"\n{'='*70}")
    print(f"🔬 SIGNAL DIAGNOSTIC FOR {symbol}")
    print(f"{'='*70}\n")
    
    # Initialize with env vars
    account_address = os.getenv('ACCOUNT_ADDRESS')
    api_secret = os.getenv('API_SECRET')
    client = HyperLiquidClient(
        account_address=account_address,
        api_key="",  # Not needed for public data
        api_secret=api_secret
    )
    strategy = SwingStrategy(symbol)
    
    # Get candles
    print("📊 Fetching market data...")
    candles = client.get_candles(symbol, interval='15m', limit=200)
    
    if not candles or len(candles) < 100:
        print(f"❌ Not enough candles: {len(candles) if candles else 0}")
        return
    
    print(f"✅ Got {len(candles)} candles")
    
    # Current price
    current_price = Decimal(str(candles[-1].get('close', candles[-1].get('c', 0))))
    print(f"💰 Current price: ${current_price:.4f}")
    
    # Calculate key indicators
    print(f"\n{'='*70}")
    print("📈 KEY INDICATORS")
    print(f"{'='*70}")
    
    prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles]
    
    # EMA200
    ema_200 = strategy._calculate_ema(prices, 200)
    ema_200_dist = ((current_price - ema_200) / ema_200 * 100) if ema_200 else 0
    print(f"EMA200: ${float(ema_200):.4f} ({'+' if ema_200_dist >= 0 else ''}{float(ema_200_dist):.2f}% from price)")
    
    if current_price > float(ema_200):
        print(f"   ✅ Price ABOVE EMA200 → LONG allowed, SHORT blocked")
    else:
        print(f"   ✅ Price BELOW EMA200 → SHORT allowed, LONG blocked")
    
    # RSI
    indicators = strategy._calculate_indicators(candles)
    rsi = indicators.get('rsi', 0)
    print(f"\nRSI: {rsi:.1f}")
    if rsi > 65:
        print(f"   ⚠️ RSI > 65 → LONG blocked (overbought)")
    elif rsi < 35:
        print(f"   ⚠️ RSI < 35 → SHORT blocked (oversold)")
    else:
        print(f"   ✅ RSI in neutral zone (35-65)")
    
    # ADX
    adx = indicators.get('adx', 0)
    print(f"\nADX: {adx:.1f}")
    if adx > 25:
        print(f"   ✅ Strong trend (ADX > 25)")
    else:
        print(f"   ⚠️ Weak trend (ADX < 25) - ranging market")
    
    # Regime Detection
    regime, confidence, params = strategy.regime_detector.detect_regime(
        candles,
        adx=indicators.get('adx'),
        atr=indicators.get('atr'),
        bb_bandwidth=indicators.get('bb_bandwidth'),
        ema_fast=indicators.get('ema_fast'),
        ema_slow=indicators.get('ema_slow')
    )
    print(f"\n🎯 Market Regime: {regime.value} ({confidence:.0%} confidence)")
    if regime == MarketRegime.TRENDING_UP:
        print(f"   ✅ LONG allowed, SHORT hard-blocked")
    elif regime == MarketRegime.TRENDING_DOWN:
        print(f"   ✅ SHORT allowed, LONG hard-blocked")
    else:
        print(f"   ✅ Both directions allowed")
    
    # Recent candles (chase detection)
    recent = candles[-5:]
    green = sum(1 for c in recent if float(c.get('close', c.get('c', 0))) > float(c.get('open', c.get('o', 0))))
    red = 5 - green
    print(f"\n🕯️ Last 5 candles: {green} green, {red} red")
    max_chase = int(os.getenv('MAX_CHASE_CANDLES', '4'))
    if green >= max_chase:
        print(f"   ⚠️ {green}+ green candles → LONG blocked (chasing top)")
    elif red >= max_chase:
        print(f"   ⚠️ {red}+ red candles → SHORT blocked (chasing bottom)")
    else:
        print(f"   ✅ No chase pattern detected")
    
    # Supertrend
    st_result = strategy.supertrend.calculate(candles)
    if st_result:
        print(f"\n📊 Supertrend: {st_result.direction.value} (strength: {st_result.strength:.1f})")
        if st_result.strength > 1.5:
            if st_result.direction.value == 'bearish':
                print(f"   ⚠️ Strong bearish → LONG hard-blocked")
            else:
                print(f"   ⚠️ Strong bullish → SHORT hard-blocked")
        else:
            print(f"   ✅ Not strong enough to hard-block")
    
    # Volume - use PREVIOUS completed candle like the fixed code
    if len(candles) >= 22:
        volumes = [c.get('volume', c.get('v', 0)) for c in candles]
        prior_volumes = sorted(volumes[-22:-2])
        median_volume = prior_volumes[len(prior_volumes) // 2]
        recent_volume = volumes[-2]  # Last COMPLETED candle
        volume_ratio = recent_volume / median_volume if median_volume > 0 else 1.0
    else:
        volume_ratio = 1.0
    min_vol = float(os.getenv('MIN_VOLUME_RATIO', '0.4'))
    print(f"\n📊 Volume: {volume_ratio:.2f}x average")
    if volume_ratio < min_vol:
        print(f"   ⚠️ Volume < {min_vol}x → ALL TRADES blocked")
    else:
        print(f"   ✅ Volume sufficient")
    
    # ATR volatility
    atr_val = indicators.get('atr')
    if atr_val:
        atr_pct = float(atr_val) / float(current_price) * 100
        min_atr = float(os.getenv('MIN_ATR_PCT', '0.2'))
        print(f"\n📊 ATR: {atr_pct:.3f}% of price")
        if atr_pct < min_atr:
            print(f"   ⚠️ ATR < {min_atr}% → ALL TRADES blocked (too quiet)")
        else:
            print(f"   ✅ Volatility sufficient")
    
    # SIGNAL SCORING
    print(f"\n{'='*70}")
    print("📊 SIGNAL SCORING")
    print(f"{'='*70}")
    
    # Get raw scores
    long_base, long_details = strategy._calculate_signal_score(
        candles, indicators, 'long', htf_candles=None, regime=regime
    )
    short_base, short_details = strategy._calculate_signal_score(
        candles, indicators, 'short', htf_candles=None, regime=regime
    )
    
    print(f"\n🟢 LONG Score: {long_base:.1f}")
    for key, val in long_details.items():
        print(f"   {key}: {val}")
    
    print(f"\n🔴 SHORT Score: {short_base:.1f}")
    for key, val in short_details.items():
        print(f"   {key}: {val}")
    
    # Threshold
    threshold = int(os.getenv('MIN_SIGNAL_SCORE', '10'))
    print(f"\n📊 Threshold: {threshold}")
    
    # HARD BLOCK SUMMARY
    print(f"\n{'='*70}")
    print("🚫 HARD BLOCK SUMMARY")
    print(f"{'='*70}")
    
    blocks = []
    
    # EMA200 block
    if current_price > float(ema_200):
        blocks.append(("SHORT", "Price above EMA200"))
    else:
        blocks.append(("LONG", "Price below EMA200"))
    
    # RSI block
    if rsi > 65:
        blocks.append(("LONG", f"RSI={rsi:.0f} > 65 (overbought)"))
    if rsi < 35:
        blocks.append(("SHORT", f"RSI={rsi:.0f} < 35 (oversold)"))
    
    # Regime block
    if regime == MarketRegime.TRENDING_UP:
        blocks.append(("SHORT", "TRENDING_UP regime"))
    elif regime == MarketRegime.TRENDING_DOWN:
        blocks.append(("LONG", "TRENDING_DOWN regime"))
    
    # Chase block
    if green >= max_chase:
        blocks.append(("LONG", f"{green}+ green candles (chasing)"))
    if red >= max_chase:
        blocks.append(("SHORT", f"{red}+ red candles (chasing)"))
    
    # Supertrend block
    if st_result and st_result.strength > 1.5:
        if st_result.direction.value == 'bearish':
            blocks.append(("LONG", f"Strong bearish Supertrend"))
        else:
            blocks.append(("SHORT", f"Strong bullish Supertrend"))
    
    # Volume/ATR block
    if volume_ratio < min_vol:
        blocks.append(("ALL", f"Volume too low ({volume_ratio:.2f}x)"))
    if atr_val and (float(atr_val) / float(current_price) * 100) < float(os.getenv('MIN_ATR_PCT', '0.2')):
        blocks.append(("ALL", f"ATR too low"))
    
    if blocks:
        for direction, reason in blocks:
            print(f"   🚫 {direction} blocked: {reason}")
    else:
        print("   ✅ No hard blocks active")
    
    # Final verdict
    print(f"\n{'='*70}")
    print("🎯 FINAL VERDICT")
    print(f"{'='*70}")
    
    long_blocked = any(b[0] in ['LONG', 'ALL'] for b in blocks)
    short_blocked = any(b[0] in ['SHORT', 'ALL'] for b in blocks)
    
    can_long = not long_blocked and long_base >= threshold
    can_short = not short_blocked and short_base >= threshold
    
    if can_long:
        print(f"   🟢 LONG: POSSIBLE (score {long_base:.0f} >= {threshold})")
    elif long_blocked:
        print(f"   🟢 LONG: BLOCKED")
    else:
        print(f"   🟢 LONG: Score too low ({long_base:.0f} < {threshold})")
    
    if can_short:
        print(f"   🔴 SHORT: POSSIBLE (score {short_base:.0f} >= {threshold})")
    elif short_blocked:
        print(f"   🔴 SHORT: BLOCKED")
    else:
        print(f"   🔴 SHORT: Score too low ({short_base:.0f} < {threshold})")
    
    if not can_long and not can_short:
        print(f"\n   ⏳ NO TRADE - waiting for better conditions")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    asyncio.run(diagnose_signals())
