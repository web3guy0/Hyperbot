#!/usr/bin/env python3
"""Detailed score diagnosis - see exactly what's scoring and what's not."""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault('BOT_MODE', 'test')

from decimal import Decimal
from app.hl.hl_client import HyperLiquidClient

def diagnose_symbol(client, symbol: str):
    print(f"\n{'='*60}")
    print(f"  {symbol} Score Diagnosis")
    print(f"{'='*60}\n")
    
    candles = client.get_candles(symbol, '5m', limit=150)
    if not candles:
        print("❌ No candle data")
        return
    
    # Calculate indicators using pandas_ta
    import pandas as pd
    import pandas_ta as ta
    
    df = pd.DataFrame(candles)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    current_price = df['close'].iloc[-1]
    
    rsi = ta.rsi(df['close'], length=14).iloc[-1]
    ema9 = ta.ema(df['close'], length=9).iloc[-1]
    ema21 = ta.ema(df['close'], length=21).iloc[-1]
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    adx = adx_df['ADX_14'].iloc[-1] if adx_df is not None else 20
    macd_df = ta.macd(df['close'])
    histogram = macd_df['MACDh_12_26_9'].iloc[-1] if macd_df is not None else 0
    
    closes = df['close'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    volumes = df['volume'].tolist()
    
    print("━━━ TECHNICAL INDICATORS ━━━")
    print(f"  Price: ${current_price:,.2f}")
    print(f"  RSI: {rsi:.1f}")
    print(f"  EMA9: {ema9:.2f}, EMA21: {ema21:.2f}")
    print(f"  ADX: {adx:.1f}")
    print(f"  MACD Histogram: {histogram:.6f}")
    
    print("\n━━━ LONG SCORE BREAKDOWN ━━━")
    score = 0
    
    # RSI (0-1)
    if rsi < 45:
        rsi_score = 1.0
        rsi_note = "✅ Oversold (<45)"
    elif rsi < 60:
        rsi_score = 0.5
        rsi_note = "✓ Neutral-bullish (<60)"
    else:
        rsi_score = 0
        rsi_note = f"❌ Overbought ({rsi:.0f} > 60)"
    score += rsi_score
    print(f"  RSI:           {rsi_score:>4.1f} pts  {rsi_note}")
    
    # EMA (0-1)
    if ema9 > ema21:
        ema_score = 1.0
        ema_note = "✅ EMA9 > EMA21"
    else:
        ema_score = 0
        ema_note = "❌ EMA9 < EMA21"
    score += ema_score
    print(f"  EMA:           {ema_score:>4.1f} pts  {ema_note}")
    
    # ADX (0-1)
    if adx > 20:
        adx_score = 1.0
        adx_note = f"✅ Trending (ADX {adx:.0f} > 20)"
    else:
        adx_score = 0.5
        adx_note = f"✓ Weak trend ({adx:.0f})"
    score += adx_score
    print(f"  ADX:           {adx_score:>4.1f} pts  {adx_note}")
    
    # MACD (0-1)
    if histogram > 0:
        macd_score = 1.0
        macd_note = "✅ Positive histogram"
    else:
        macd_score = 0
        macd_note = "❌ Negative histogram"
    score += macd_score
    print(f"  MACD:          {macd_score:>4.1f} pts  {macd_note}")
    
    print(f"\n  Technical subtotal: {score:.1f}/4")
    
    # SMC Analysis (0-2)
    print("\n━━━ SMART MONEY CONCEPTS ━━━")
    try:
        from app.strategies.smc.smc_analyzer import SMCAnalyzer
        smc = SMCAnalyzer()
        smc_result = smc.analyze(candles, Decimal(str(current_price)))
        bias = smc_result.get('bias', 'neutral')
        signals = smc_result.get('signals', [])
        
        smc_score = 0
        if bias == 'bullish':
            smc_score += 1
            print(f"  Bias:          +1.0 pts  ✅ Bullish")
        else:
            print(f"  Bias:          +0.0 pts  ❌ {bias}")
        
        for sig in signals[:2]:  # First 2 signals
            if sig.get('direction') == 'long':
                sig_type = sig.get('type', '')
                if sig_type == 'liquidity_sweep':
                    smc_score += 1
                    print(f"  Signal:        +1.0 pts  ✅ {sig_type}")
                elif sig_type in ['fvg_fill', 'order_block']:
                    smc_score += 0.5
                    print(f"  Signal:        +0.5 pts  ✅ {sig_type}")
        
        smc_score = min(smc_score, 2)
        score += smc_score
        print(f"\n  SMC subtotal: {smc_score:.1f}/2")
    except Exception as e:
        print(f"  ⚠️ SMC error: {e}")
    
    # HTF Analysis (0-2)
    print("\n━━━ HTF ALIGNMENT ━━━")
    try:
        from app.strategies.multi_timeframe.mtf_analyzer import MTFAnalyzer
        mtf = MTFAnalyzer(client, symbol)
        htf_score_raw, htf_reason = mtf.get_alignment_score('long')
        htf_score = htf_score_raw * 2  # Scale 0-1 to 0-2
        score += htf_score
        print(f"  HTF:           {htf_score:.1f} pts  ({htf_reason})")
    except Exception as e:
        print(f"  ⚠️ HTF error: {e}")
    
    # Order Flow (0-2)
    print("\n━━━ ORDER FLOW ━━━")
    try:
        from app.strategies.order_flow.order_flow_analyzer import OrderFlowAnalyzer
        of = OrderFlowAnalyzer(client, symbol)
        of_result = of.analyze(candles)
        
        of_score = 0
        of_bias = of_result.get('bias', 'neutral')
        if of_bias == 'bullish':
            of_score += 1
            print(f"  Bias:          +1.0 pts  ✅ Bullish")
        else:
            print(f"  Bias:          +0.0 pts  ❌ {of_bias}")
        
        poc_dist = of_result.get('poc_distance_pct')
        if poc_dist is not None and abs(poc_dist) < 0.5:
            of_score += 1
            print(f"  POC:           +1.0 pts  ✅ Near POC ({poc_dist:.2f}%)")
        else:
            print(f"  POC:           +0.0 pts  ❌ Far from POC")
        
        score += of_score
        print(f"\n  Order Flow subtotal: {of_score:.1f}/2")
    except Exception as e:
        print(f"  ⚠️ Order Flow error: {e}")
    
    # Break of Structure (0-1.5)
    print("\n━━━ BREAK OF STRUCTURE ━━━")
    try:
        smc = SMCAnalyzer()
        bos_score, bos_reason = smc.get_bos_signal('long')
        score += bos_score
        if bos_score > 0:
            print(f"  BoS:           +{bos_score:.1f} pts  ✅ {bos_reason}")
        elif bos_score < 0:
            print(f"  BoS:           {bos_score:.1f} pts  ❌ {bos_reason}")
        else:
            print(f"  BoS:           +0.0 pts  ⚪ No BoS signal")
    except Exception as e:
        print(f"  ⚠️ BoS error: {e}")
    
    print(f"\n{'='*40}")
    print(f"  BASE SCORE: {score:.1f}/12 (before enhancements)")
    print(f"{'='*40}")
    
    # Enhanced scoring
    print("\n━━━ ENHANCED SCORING ━━━")
    
    # Supertrend
    try:
        from app.strategies.indicators.supertrend import Supertrend
        st = Supertrend()
        st_result = st.calculate(candles)
        if st_result:
            if st_result.direction.value == 'bullish':
                st_score = 2.0
                print(f"  Supertrend:    +2.0 pts  ✅ BULLISH")
            else:
                st_score = -1.5
                print(f"  Supertrend:    -1.5 pts  ❌ BEARISH")
            score += st_score
        else:
            print(f"  Supertrend:    +0.0 pts  ⚪ No data")
    except Exception as e:
        print(f"  ⚠️ Supertrend error: {e}")
    
    # Volume
    avg_vol = sum(volumes[-20:]) / 20
    curr_vol = volumes[-1]
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio > 1.2:
        vol_score = 1.0
        print(f"  Volume:        +1.0 pts  ✅ Above avg ({vol_ratio:.1f}x)")
    elif vol_ratio < 0.6:
        vol_score = -0.5
        print(f"  Volume:        -0.5 pts  ❌ Weak ({vol_ratio:.1f}x)")
    else:
        vol_score = 0
        print(f"  Volume:        +0.0 pts  ⚪ Normal ({vol_ratio:.1f}x)")
    score += vol_score
    
    # Regime
    try:
        from app.strategies.adaptive.market_regime import MarketRegimeDetector, MarketRegime
        regime_det = MarketRegimeDetector()
        regime, conf, _ = regime_det.detect_regime(candles)
        
        if regime == MarketRegime.TRENDING_UP:
            regime_score = 2.0
            print(f"  Regime:        +2.0 pts  ✅ TRENDING UP ({conf:.0f}%)")
        elif regime == MarketRegime.TRENDING_DOWN:
            regime_score = -5.0  # Counter-trend penalty
            print(f"  Regime:        -5.0 pts  ❌ TRENDING DOWN (counter-trend!)")
        else:
            regime_score = 0
            print(f"  Regime:        +0.0 pts  ⚪ {regime.value} ({conf:.0f}%)")
        score += regime_score
    except Exception as e:
        print(f"  ⚠️ Regime error: {e}")
    
    print(f"\n{'='*60}")
    print(f"  FINAL LONG SCORE: {int(score)}/25")
    print(f"  THRESHOLD: 8/25")
    if score >= 8:
        print(f"  ✅ SIGNAL WOULD TRIGGER")
    else:
        print(f"  ❌ NO SIGNAL (need +{8-int(score)} more points)")
    print(f"{'='*60}")


if __name__ == "__main__":
    account = os.environ.get('ACCOUNT_ADDRESS', '')
    api_secret = os.environ.get('API_SECRET', '')
    client = HyperLiquidClient(account, api_secret, api_secret)
    
    for symbol in ['BTC', 'ETH', 'SOL']:
        diagnose_symbol(client, symbol)
