#!/usr/bin/env python3
"""
Live Signal Test - Tests signal generation in actual bot context
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# Colors
G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; B = '\033[94m'; M = '\033[95m'; C = '\033[96m'; E = '\033[0m'
def ok(m): print(f"{G}✅ {m}{E}")
def err(m): print(f"{R}❌ {m}{E}")
def warn(m): print(f"{Y}⚠️  {m}{E}")
def info(m): print(f"{B}ℹ️  {m}{E}")
def hdr(m): print(f"\n{C}{'='*60}\n  {m}\n{'='*60}{E}\n")

async def test_live_signals():
    hdr("LIVE SIGNAL GENERATION TEST")
    
    # Initialize HyperAIBot like the real bot does
    from app.bot import HyperAIBot
    
    bot = HyperAIBot()
    info("Created HyperAIBot instance")
    
    # Initialize components
    await bot.initialize()
    ok("Bot initialized")
    
    # Wait a moment for WebSocket to connect
    await asyncio.sleep(3)
    
    # Get account state
    account_state = await bot.client.get_account_state()
    if account_state:
        ok("Got account state")
        margin_summary = account_state.get('marginSummary', {})
        account_value = float(margin_summary.get('accountValue', 0))
        info(f"Account value: ${account_value:,.2f}")
    else:
        err("No account state")
        return
    
    # Check positions
    positions = account_state.get('positions', [])
    open_positions = [p for p in positions if float(p.get('size', 0)) != 0]
    info(f"Open positions: {len(open_positions)}")
    
    # Test multi-asset manager
    hdr("Multi-Asset Manager Status")
    if bot.asset_manager:
        ok("Asset manager exists")
        
        # Update from account state
        bot.asset_manager.update_from_account_state(account_state)
        
        for symbol, state in bot.asset_manager.assets.items():
            status = "🟢 Available" if not state.has_position else "🔴 Has Position"
            print(f"  {symbol}: {status}, Enabled={state.is_enabled}")
        
        # Check what's available
        available = bot.asset_manager.get_assets_without_positions()
        info(f"Available for trading: {available}")
        
        can_open = bot.asset_manager.can_open_new_position()
        info(f"Can open new position: {can_open}")
    else:
        err("No asset manager")
    
    # Test signal generation for each asset
    hdr("Testing Signal Generation")
    
    for symbol in ['SOL', 'ETH', 'BTC']:
        print(f"\n  {M}--- {symbol} ---{E}")
        
        # Get strategy
        strategy = bot.strategies.get(symbol)
        if not strategy:
            err(f"  No strategy for {symbol}")
            continue
        ok(f"  Strategy: {strategy.__class__.__name__}")
        
        # Fetch candles
        candles = bot.client.get_candles(symbol, bot.timeframe, 150)
        if not candles:
            err(f"  No candles for {symbol}")
            continue
        ok(f"  Got {len(candles)} candles")
        
        # Check candle format
        sample = candles[0]
        info(f"  Candle format: {list(sample.keys())}")
        
        # Build market_data
        market_data = bot.websocket.get_market_data(symbol) or {}
        market_data['candles'] = candles
        
        current_price = market_data.get('price', candles[-1].get('close', 0))
        info(f"  Current price: ${float(current_price):,.2f}")
        
        # Try to generate signal
        try:
            signal = await strategy.generate_signal(market_data, account_state)
            
            if signal:
                ok(f"  SIGNAL: {signal.get('signal_type', 'unknown')}")
                info(f"    Direction: {signal.get('direction', 'unknown')}")
                info(f"    Confidence: {signal.get('confidence', 0):.1%}")
                info(f"    Entry: ${signal.get('entry', 0):,.2f}")
            else:
                warn(f"  No signal generated")
                
        except Exception as e:
            err(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Check the last few score logs
    hdr("Signal Score Analysis")
    
    # Manually test score calculation for one asset
    symbol = "BTC"
    strategy = bot.strategies.get(symbol)
    candles = bot.client.get_candles(symbol, bot.timeframe, 150)
    
    if strategy and candles:
        # Get indicators
        indicators = strategy._calculate_indicators(candles)
        
        if indicators:
            ok(f"Calculated indicators for {symbol}")
            
            # Key indicators
            print(f"\n  Key Indicators:")
            print(f"    RSI: {indicators.get('rsi', 'N/A')}")
            print(f"    EMA Fast: {indicators.get('ema_fast', 'N/A')}")
            print(f"    EMA Slow: {indicators.get('ema_slow', 'N/A')}")
            print(f"    MACD Histogram: {indicators.get('macd', {}).get('histogram', 'N/A')}")
            print(f"    ADX: {indicators.get('adx', 'N/A')}")
            
            # Get regime
            regime, confidence, params = strategy.regime_detector.detect_regime(
                candles,
                adx=indicators.get('adx'),
                atr=indicators.get('atr'),
                bb_bandwidth=indicators.get('bb_bandwidth'),
                ema_fast=indicators.get('ema_fast'),
                ema_slow=indicators.get('ema_slow'),
            )
            print(f"\n  Market Regime: {regime.value} (confidence: {confidence:.1%})")
        else:
            err(f"Failed to calculate indicators")
    
    # Cleanup
    hdr("Cleanup")
    await bot.shutdown()
    ok("Bot shutdown complete")
    
    hdr("Summary")
    print("""
  If you see "No signal generated" for all assets, check:
  
  1. SCORES TOO LOW: Signal threshold is 15/25 (60%)
     - Markets may be in consolidation
     - Try lowering MIN_SIGNAL_SCORE in .env
  
  2. REGIME BLOCKING: Counter-trend trades are blocked
     - Cannot LONG in TRENDING_DOWN
     - Cannot SHORT in TRENDING_UP
  
  3. COOLDOWN ACTIVE: After each signal, 15min cooldown per asset
     - Check asset manager cooldown status
  
  4. CONFIRMATION NOT MET: Requires 3 consecutive matching scans
     - Signal must appear 3 times in a row
  """)

if __name__ == "__main__":
    asyncio.run(test_live_signals())
