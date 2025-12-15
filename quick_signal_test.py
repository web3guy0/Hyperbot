#!/usr/bin/env python3
"""
Quick Signal Test - Minimal test of signal generation
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from decimal import Decimal

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# Suppress logging
import logging
logging.basicConfig(level=logging.WARNING)
for name in ['app', 'HyperBot', 'websocket']:
    logging.getLogger(name).setLevel(logging.WARNING)

# Colors
G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; B = '\033[94m'; M = '\033[95m'; C = '\033[96m'; E = '\033[0m'

def ok(m): print(f"{G}✅ {m}{E}")
def err(m): print(f"{R}❌ {m}{E}")
def warn(m): print(f"{Y}⚠️  {m}{E}")
def info(m): print(f"{B}ℹ️  {m}{E}")
def hdr(m): print(f"\n{C}{'='*60}\n  {m}\n{'='*60}{E}\n")

async def main():
    hdr("QUICK SIGNAL GENERATION TEST")
    
    # Initialize client directly
    from app.hl.hl_client import HyperLiquidClient
    
    account = os.getenv('ACCOUNT_ADDRESS')
    secret = os.getenv('API_SECRET')
    
    client = HyperLiquidClient(account, secret, secret, testnet=False)
    ok("Client connected")
    
    # Get account state
    account_state = await client.get_account_state()
    if account_state:
        margin = account_state.get('marginSummary', {})
        value = float(margin.get('accountValue', 0))
        info(f"Account: ${value:,.2f}")
    
    # Check positions
    positions = account_state.get('positions', []) if account_state else []
    open_positions = [p for p in positions if float(p.get('size', 0)) != 0]
    info(f"Open positions: {len(open_positions)}")
    for p in open_positions:
        print(f"  • {p['symbol']}: {p.get('size')} @ ${p.get('entry_price')}")
    
    # Test each asset
    hdr("Signal Generation Test")
    
    from app.strategies.rule_based.swing_strategy import SwingStrategy
    from app.strategies.adaptive.market_regime import MarketRegime
    
    test_assets = ['BTC', 'ETH', 'SOL']
    
    for symbol in test_assets:
        print(f"\n{M}━━━ {symbol} ━━━{E}")
        
        # Fetch candles
        candles = client.get_candles(symbol, '1m', 150)
        if not candles or len(candles) < 100:
            err(f"Not enough candles: {len(candles) if candles else 0}")
            continue
        
        ok(f"Candles: {len(candles)}")
        current_price = float(candles[-1]['close'])
        info(f"Price: ${current_price:,.2f}")
        
        # Initialize strategy
        strategy = SwingStrategy(symbol=symbol)
        
        # Calculate indicators
        indicators = strategy._calculate_indicators(candles)
        if not indicators:
            err("Failed to calculate indicators")
            continue
        
        # Key indicator values
        rsi = indicators.get('rsi', 0)
        ema_fast = indicators.get('ema_fast', 0)
        ema_slow = indicators.get('ema_slow', 0)
        adx = indicators.get('adx', 0)
        atr = indicators.get('atr', 0)
        
        print(f"  RSI: {rsi:.1f}")
        print(f"  EMA9/21: {ema_fast:.2f} / {ema_slow:.2f}")
        print(f"  ADX: {adx:.1f}")
        print(f"  ATR: {atr:.2f}")
        
        # Detect regime
        regime, confidence, params = strategy.regime_detector.detect_regime(
            candles,
            adx=adx,
            atr=atr,
            bb_bandwidth=indicators.get('bb_bandwidth'),
            ema_fast=ema_fast,
            ema_slow=ema_slow,
        )
        print(f"  Regime: {M}{regime.value}{E} (conf: {confidence:.0%})")
        
        # Calculate scores
        long_score = strategy._calculate_signal_score(
            direction='long',
            indicators=indicators,
            regime=regime,
            regime_params=params,
            smc_analysis={},
            of_analysis={},
            current_price=Decimal(str(current_price)),
        )
        
        short_score = strategy._calculate_signal_score(
            direction='short',
            indicators=indicators,
            regime=regime,
            regime_params=params,
            smc_analysis={},
            of_analysis={},
            current_price=Decimal(str(current_price)),
        )
        
        # Enhanced scores
        long_enhanced, _ = strategy._calculate_enhanced_score('long', candles, indicators, long_score)
        short_enhanced, _ = strategy._calculate_enhanced_score('short', candles, indicators, short_score)
        
        threshold = strategy.min_signal_score
        max_score = strategy.max_signal_score
        
        # Display scores
        print(f"\n  {G}LONG:  {long_enhanced:2.0f}/{max_score} {'█' * int(long_enhanced/max_score*20) + '░' * (20-int(long_enhanced/max_score*20))}{E}")
        print(f"  {R}SHORT: {short_enhanced:2.0f}/{max_score} {'█' * int(short_enhanced/max_score*20) + '░' * (20-int(short_enhanced/max_score*20))}{E}")
        print(f"  {Y}THRESHOLD: {threshold}/{max_score}{E}")
        
        # Signal outcome
        if long_enhanced >= threshold:
            if regime == MarketRegime.TRENDING_DOWN:
                warn(f"  LONG score OK but BLOCKED (counter-trend)")
            else:
                ok(f"  ✓ LONG SIGNAL would generate!")
        elif short_enhanced >= threshold:
            if regime == MarketRegime.TRENDING_UP:
                warn(f"  SHORT score OK but BLOCKED (counter-trend)")
            else:
                ok(f"  ✓ SHORT SIGNAL would generate!")
        else:
            gap_long = threshold - long_enhanced
            gap_short = threshold - short_enhanced
            warn(f"  No signal: need +{min(gap_long, gap_short):.0f} more points")
    
    hdr("Recommendations")
    print(f"""
  Current threshold: {threshold}/25 ({threshold/25*100:.0f}%)
  
  If scores are consistently below threshold:
  
  1. {Y}Lower threshold{E}: Add to .env:
     MIN_SIGNAL_SCORE=12  (or 10 for more signals)
  
  2. {Y}Reduce confirmation{E}: Add to .env:
     SIGNAL_CONFIRMATION_SCANS=2  (from 3)
  
  3. {Y}Check market conditions{E}:
     - Consolidation = low scores (normal)
     - Wait for trending conditions
  
  4. {Y}Current settings{E} (from .env):
     - Signal Threshold: {os.getenv('MIN_SIGNAL_SCORE', '15')}/25
     - Confirmation Scans: {os.getenv('SIGNAL_CONFIRMATION_SCANS', '3')}
     - Direction Lock: {os.getenv('DIRECTION_LOCK_SECONDS', '900')}s
    """)

if __name__ == "__main__":
    asyncio.run(main())
