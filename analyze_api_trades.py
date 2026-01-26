#!/usr/bin/env python3
"""
Analyze trading performance directly from Hyperliquid API.
Properly pairs open/close fills into complete round-trip trades.
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperliquid.info import Info
from hyperliquid.utils import constants


def fetch_fills_from_api(address: str, testnet: bool = False) -> List[Dict]:
    """Fetch all fills from Hyperliquid API."""
    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(base_url, skip_ws=True)
    
    fills = info.user_fills(address)
    return fills


def pair_fills_into_trades(fills: List[Dict], symbol_filter: Optional[str] = None) -> List[Dict]:
    """
    Pair fills into complete round-trip trades.
    
    A trade consists of:
    - Opening fill(s): Increasing position
    - Closing fill(s): Decreasing/closing position (has closedPnl)
    
    Returns list of completed trades with full P&L data.
    """
    # Sort by time ascending
    fills = sorted(fills, key=lambda x: x['time'])
    
    # Filter by symbol if specified
    if symbol_filter:
        fills = [f for f in fills if f.get('coin') == symbol_filter]
    
    # Group by coin
    by_coin = defaultdict(list)
    for f in fills:
        by_coin[f.get('coin', 'UNKNOWN')].append(f)
    
    completed_trades = []
    
    for coin, coin_fills in by_coin.items():
        # Track position state
        position_size = Decimal('0')
        position_side = None  # 'long' or 'short'
        entry_fills = []
        entry_value = Decimal('0')
        entry_time = None
        
        for fill in coin_fills:
            side = fill.get('side')  # 'A' = sell/ask, 'B' = buy/bid
            size = Decimal(str(fill.get('sz', '0')))
            price = Decimal(str(fill.get('px', '0')))
            closed_pnl = Decimal(str(fill.get('closedPnl', '0')))
            fee = Decimal(str(fill.get('fee', '0')))
            fill_time = datetime.fromtimestamp(fill['time'] / 1000, tz=timezone.utc)
            
            is_buy = side == 'B'
            
            # Determine if this is opening or closing
            if position_size == 0:
                # Opening new position
                position_side = 'long' if is_buy else 'short'
                position_size = size
                entry_fills = [fill]
                entry_value = size * price
                entry_time = fill_time
                
            elif (position_side == 'long' and is_buy) or (position_side == 'short' and not is_buy):
                # Adding to position (same direction)
                entry_fills.append(fill)
                entry_value += size * price
                position_size += size
                
            else:
                # Closing position (opposite direction)
                close_size = min(size, position_size)
                
                if close_size > 0:
                    # Calculate trade metrics
                    avg_entry = entry_value / position_size if position_size > 0 else price
                    
                    trade = {
                        'coin': coin,
                        'side': position_side,
                        'entry_time': entry_time,
                        'exit_time': fill_time,
                        'duration_seconds': int((fill_time - entry_time).total_seconds()) if entry_time else 0,
                        'entry_price': float(avg_entry),
                        'exit_price': float(price),
                        'size': float(close_size),
                        'notional': float(close_size * price),
                        'pnl': float(closed_pnl),
                        'fee': float(fee),
                        'net_pnl': float(closed_pnl - fee),
                        'pnl_pct': float(closed_pnl / (close_size * avg_entry) * 100) if avg_entry > 0 else 0,
                        'entry_fills': len(entry_fills),
                        'oid': fill.get('oid'),
                        'hash': fill.get('hash'),
                    }
                    completed_trades.append(trade)
                
                # Update position
                position_size -= close_size
                remaining = size - close_size
                
                if position_size == 0:
                    entry_fills = []
                    entry_value = Decimal('0')
                    entry_time = None
                    position_side = None
                    
                    # If there's remaining size, it's opening a new position in opposite direction
                    if remaining > 0:
                        position_side = 'long' if is_buy else 'short'
                        position_size = remaining
                        entry_fills = [fill]
                        entry_value = remaining * price
                        entry_time = fill_time
                else:
                    # Partial close - adjust entry value proportionally
                    entry_value = entry_value * (position_size / (position_size + close_size))
    
    # Sort by exit time descending (most recent first)
    completed_trades.sort(key=lambda x: x['exit_time'], reverse=True)
    
    return completed_trades


def analyze_trades(trades: List[Dict], limit: int = 100):
    """Comprehensive trade analysis."""
    
    # Take last N trades
    trades = trades[:limit]
    
    if not trades:
        print("❌ No completed trades found!")
        return
    
    print()
    print("=" * 75)
    print(f"📊 HYPERLIQUID API - TRADE PERFORMANCE ANALYSIS")
    print(f"   Last {len(trades)} Round-Trip Trades")
    print("=" * 75)
    
    # ===== OVERALL PERFORMANCE =====
    total_pnl = sum(t['pnl'] for t in trades)
    total_fees = sum(t['fee'] for t in trades)
    net_pnl = sum(t['net_pnl'] for t in trades)
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    breakeven = [t for t in trades if t['pnl'] == 0]
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    print()
    print("📈 OVERALL PERFORMANCE")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Gross P&L: ${total_pnl:+.2f}")
    print(f"   Total Fees: ${total_fees:.2f}")
    print(f"   Net P&L: ${net_pnl:+.2f}")
    print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakeven)}BE)")
    
    if wins:
        avg_win = sum(t['pnl'] for t in wins) / len(wins)
        avg_win_pct = sum(t['pnl_pct'] for t in wins) / len(wins)
        best_win = max(t['pnl'] for t in wins)
        best_win_pct = max(t['pnl_pct'] for t in wins)
        print()
        print("   ✅ Wins:")
        print(f"      Average Win: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
        print(f"      Best Win: ${best_win:.2f} ({best_win_pct:.2f}%)")
    
    if losses:
        avg_loss = sum(t['pnl'] for t in losses) / len(losses)
        avg_loss_pct = sum(t['pnl_pct'] for t in losses) / len(losses)
        worst_loss = min(t['pnl'] for t in losses)
        worst_loss_pct = min(t['pnl_pct'] for t in losses)
        print()
        print("   ❌ Losses:")
        print(f"      Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
        print(f"      Worst Loss: ${worst_loss:.2f} ({worst_loss_pct:.2f}%)")
    
    # ===== RISK METRICS =====
    print()
    print("─" * 75)
    print("📊 RISK METRICS")
    
    if wins and losses:
        avg_win_amt = sum(t['pnl'] for t in wins) / len(wins)
        avg_loss_amt = abs(sum(t['pnl'] for t in losses) / len(losses))
        rr_ratio = avg_win_amt / avg_loss_amt if avg_loss_amt > 0 else float('inf')
        print(f"   Risk/Reward Ratio: {rr_ratio:.2f}")
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(f"   Profit Factor: {profit_factor:.2f}")
        
        # Expectancy
        expectancy = total_pnl / len(trades)
        print(f"   Expectancy: ${expectancy:.3f}/trade")
        
        # Edge calculation: (Win% * Avg Win) - (Loss% * Avg Loss)
        edge = (len(wins)/len(trades) * avg_win_amt) - (len(losses)/len(trades) * avg_loss_amt)
        print(f"   Edge: ${edge:.3f}/trade")
    
    # ===== BY DIRECTION =====
    print()
    print("─" * 75)
    print("📊 BY DIRECTION")
    
    longs = [t for t in trades if t['side'] == 'long']
    shorts = [t for t in trades if t['side'] == 'short']
    
    for direction, dtrades in [('LONG 📈', longs), ('SHORT 📉', shorts)]:
        if dtrades:
            d_pnl = sum(t['pnl'] for t in dtrades)
            d_wins = len([t for t in dtrades if t['pnl'] > 0])
            d_wr = d_wins / len(dtrades) * 100
            d_avg_pnl = d_pnl / len(dtrades)
            print(f"   {direction}: {len(dtrades):3} trades | P&L: ${d_pnl:+8.2f} | WR: {d_wr:5.1f}% | Avg: ${d_avg_pnl:+.2f}")
    
    # ===== DURATION ANALYSIS =====
    print()
    print("─" * 75)
    print("⏱️  TRADE DURATION")
    
    durations = [t['duration_seconds'] for t in trades if t['duration_seconds'] > 0]
    if durations:
        avg_dur = sum(durations) / len(durations)
        min_dur = min(durations)
        max_dur = max(durations)
        
        def format_dur(secs):
            if secs < 60:
                return f"{secs}s"
            elif secs < 3600:
                return f"{secs//60}m {secs%60}s"
            else:
                return f"{secs//3600}h {(secs%3600)//60}m"
        
        print(f"   Average: {format_dur(int(avg_dur))}")
        print(f"   Shortest: {format_dur(min_dur)}")
        print(f"   Longest: {format_dur(max_dur)}")
        
        # Win vs Loss duration
        win_durs = [t['duration_seconds'] for t in wins if t['duration_seconds'] > 0]
        loss_durs = [t['duration_seconds'] for t in losses if t['duration_seconds'] > 0]
        if win_durs:
            print(f"   Avg Win Duration: {format_dur(int(sum(win_durs)/len(win_durs)))}")
        if loss_durs:
            print(f"   Avg Loss Duration: {format_dur(int(sum(loss_durs)/len(loss_durs)))}")
    
    # ===== SIZE ANALYSIS =====
    print()
    print("─" * 75)
    print("📏 POSITION SIZE ANALYSIS")
    
    sizes = [t['notional'] for t in trades]
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        print(f"   Average Notional: ${avg_size:.2f}")
        print(f"   Min: ${min_size:.2f} | Max: ${max_size:.2f}")
        
        # Size vs outcome
        big_trades = [t for t in trades if t['notional'] > avg_size]
        small_trades = [t for t in trades if t['notional'] <= avg_size]
        
        if big_trades:
            big_pnl = sum(t['pnl'] for t in big_trades)
            big_wr = len([t for t in big_trades if t['pnl'] > 0]) / len(big_trades) * 100
            print(f"   Above-avg size: {len(big_trades)} trades, ${big_pnl:+.2f}, {big_wr:.0f}% WR")
        if small_trades:
            small_pnl = sum(t['pnl'] for t in small_trades)
            small_wr = len([t for t in small_trades if t['pnl'] > 0]) / len(small_trades) * 100
            print(f"   Below-avg size: {len(small_trades)} trades, ${small_pnl:+.2f}, {small_wr:.0f}% WR")
    
    # ===== P&L DISTRIBUTION =====
    print()
    print("─" * 75)
    print("📊 P&L DISTRIBUTION")
    
    pnl_ranges = [
        ("> $1.00", lambda p: p > 1.0),
        ("$0.50 - $1.00", lambda p: 0.5 <= p <= 1.0),
        ("$0.10 - $0.50", lambda p: 0.1 <= p < 0.5),
        ("$0 - $0.10", lambda p: 0 < p < 0.1),
        ("$0 (BE)", lambda p: p == 0),
        ("-$0.10 - $0", lambda p: -0.1 < p < 0),
        ("-$0.50 - -$0.10", lambda p: -0.5 <= p <= -0.1),
        ("< -$0.50", lambda p: p < -0.5),
    ]
    
    for label, check in pnl_ranges:
        count = len([t for t in trades if check(t['pnl'])])
        if count > 0:
            pct = count / len(trades) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {label:20} {bar} {count:3} ({pct:5.1f}%)")
    
    # ===== RECENT TRADES =====
    print()
    print("─" * 75)
    print("📋 LAST 20 TRADES (Most Recent First)")
    print("─" * 75)
    print(f"   {'Dir':4} {'Coin':6} {'P&L':>10} {'%':>8} {'Size':>8} {'Duration':>10} {'Exit Time'}")
    print("   " + "-" * 70)
    
    for t in trades[:20]:
        emoji = '✅' if t['pnl'] > 0 else ('❌' if t['pnl'] < 0 else '⚪')
        dir_str = '🟢L' if t['side'] == 'long' else '🔴S'
        dur = t['duration_seconds']
        if dur >= 3600:
            dur_str = f"{dur//3600}h{(dur%3600)//60}m"
        elif dur >= 60:
            dur_str = f"{dur//60}m{dur%60}s"
        else:
            dur_str = f"{dur}s"
        exit_time = t['exit_time'].strftime('%m/%d %H:%M')
        
        print(f"   {emoji}{dir_str} {t['coin']:6} ${t['pnl']:+8.2f} {t['pnl_pct']:+7.2f}% ${t['notional']:7.0f} {dur_str:>10} {exit_time}")
    
    # ===== TIME ANALYSIS =====
    print()
    print("─" * 75)
    print("📅 TIME ANALYSIS")
    
    if trades:
        oldest = min(t['entry_time'] for t in trades if t['entry_time'])
        newest = max(t['exit_time'] for t in trades)
        print(f"   From: {oldest.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   To:   {newest.strftime('%Y-%m-%d %H:%M')} UTC")
        
        days = (newest - oldest).days + 1
        hours = (newest - oldest).total_seconds() / 3600
        print(f"   Span: {days} days ({hours:.0f} hours)")
        
        if days > 0:
            print(f"   Trades/Day: {len(trades)/days:.1f}")
            print(f"   P&L/Day: ${total_pnl/days:+.2f}")
        
        # By hour of day
        print()
        print("   📊 Performance by Hour (UTC):")
        by_hour = defaultdict(list)
        for t in trades:
            hour = t['exit_time'].hour
            by_hour[hour].append(t['pnl'])
        
        # Find best/worst hours
        hour_stats = [(h, sum(pnls), len(pnls)) for h, pnls in by_hour.items()]
        hour_stats.sort(key=lambda x: x[1], reverse=True)
        
        if hour_stats:
            best_hour = hour_stats[0]
            worst_hour = hour_stats[-1]
            print(f"      Best Hour: {best_hour[0]:02d}:00 UTC | ${best_hour[1]:+.2f} ({best_hour[2]} trades)")
            print(f"      Worst Hour: {worst_hour[0]:02d}:00 UTC | ${worst_hour[1]:+.2f} ({worst_hour[2]} trades)")
    
    # ===== STREAKS =====
    print()
    print("─" * 75)
    print("🔥 STREAKS")
    
    # Sort chronologically for streak analysis
    sorted_trades = sorted(trades, key=lambda t: t['exit_time'])
    
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0
    
    max_win_streak_pnl = 0
    max_loss_streak_pnl = 0
    current_win_pnl = 0
    current_loss_pnl = 0
    
    for t in sorted_trades:
        if t['pnl'] > 0:
            current_win += 1
            current_win_pnl += t['pnl']
            if current_win > max_win_streak:
                max_win_streak = current_win
                max_win_streak_pnl = current_win_pnl
            current_loss = 0
            current_loss_pnl = 0
        elif t['pnl'] < 0:
            current_loss += 1
            current_loss_pnl += t['pnl']
            if current_loss > max_loss_streak:
                max_loss_streak = current_loss
                max_loss_streak_pnl = current_loss_pnl
            current_win = 0
            current_win_pnl = 0
        else:
            current_win = 0
            current_loss = 0
            current_win_pnl = 0
            current_loss_pnl = 0
    
    print(f"   Max Win Streak: {max_win_streak} trades (${max_win_streak_pnl:+.2f})")
    print(f"   Max Loss Streak: {max_loss_streak} trades (${max_loss_streak_pnl:+.2f})")
    
    # Current streak
    current_streak = 0
    streak_type = None
    streak_pnl = 0
    for t in reversed(sorted_trades):
        if streak_type is None:
            streak_type = 'win' if t['pnl'] > 0 else ('loss' if t['pnl'] < 0 else None)
            if streak_type:
                current_streak = 1
                streak_pnl = t['pnl']
        elif streak_type == 'win' and t['pnl'] > 0:
            current_streak += 1
            streak_pnl += t['pnl']
        elif streak_type == 'loss' and t['pnl'] < 0:
            current_streak += 1
            streak_pnl += t['pnl']
        else:
            break
    
    if current_streak > 0:
        emoji = '🔥' if streak_type == 'win' else '💀'
        print(f"   Current Streak: {current_streak} {streak_type}s {emoji} (${streak_pnl:+.2f})")
    
    # ===== DRAWDOWN =====
    print()
    print("─" * 75)
    print("📉 EQUITY CURVE & DRAWDOWN")
    
    running_pnl = 0
    peak = 0
    max_dd = 0
    max_dd_start = None
    max_dd_end = None
    dd_start = None
    
    equity_curve = []
    for t in sorted_trades:
        running_pnl += t['pnl']
        equity_curve.append((t['exit_time'], running_pnl))
        
        if running_pnl > peak:
            peak = running_pnl
            dd_start = t['exit_time']
        
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd
            max_dd_start = dd_start
            max_dd_end = t['exit_time']
    
    print(f"   Starting: $0.00")
    print(f"   Peak: ${peak:+.2f}")
    print(f"   Final: ${running_pnl:+.2f}")
    print(f"   Max Drawdown: ${max_dd:.2f}")
    if max_dd_start and max_dd_end:
        print(f"   Drawdown Period: {max_dd_start.strftime('%m/%d %H:%M')} - {max_dd_end.strftime('%m/%d %H:%M')}")
    
    # Mini equity curve visualization
    print()
    print("   Equity Curve (last 50 trades):")
    curve_data = equity_curve[-50:]
    if curve_data:
        min_eq = min(e[1] for e in curve_data)
        max_eq = max(e[1] for e in curve_data)
        eq_range = max_eq - min_eq if max_eq != min_eq else 1
        
        line = "   "
        for _, eq in curve_data:
            normalized = (eq - min_eq) / eq_range
            if normalized > 0.8:
                line += "▁"
            elif normalized > 0.6:
                line += "▂"
            elif normalized > 0.4:
                line += "▃"
            elif normalized > 0.2:
                line += "▄"
            else:
                line += "▅"
        print(line)
        print(f"   ${min_eq:+.2f}" + " " * (len(curve_data) - 10) + f"${max_eq:+.2f}")
    
    # ===== VERDICT =====
    print()
    print("=" * 75)
    print("🎯 VERDICT & INSIGHTS")
    print("=" * 75)
    
    # Calculate key metrics for verdict
    pf = profit_factor if 'profit_factor' in dir() else (sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else float('inf'))
    rr = rr_ratio if 'rr_ratio' in dir() else (sum(t['pnl'] for t in wins) / len(wins)) / (abs(sum(t['pnl'] for t in losses)) / len(losses)) if wins and losses else 0
    
    print()
    if pf >= 1.5 and win_rate >= 50:
        print("   ✅ PROFITABLE STRATEGY")
        print("      Good profit factor and solid win rate!")
    elif pf >= 1.2 and win_rate >= 45:
        print("   ⚠️ MARGINALLY PROFITABLE")
        print("      Working but has room for improvement")
    elif pf >= 1.0:
        print("   ⚠️ BREAK-EVEN")
        print("      Covering costs but not making money")
    else:
        print("   ❌ LOSING STRATEGY")
        print("      Needs significant adjustments")
    
    print()
    print("   💡 Key Insights:")
    
    # Win rate insight
    if win_rate >= 70:
        print("      ✅ Excellent win rate - entry timing is good")
    elif win_rate >= 55:
        print("      ✅ Solid win rate")
    elif win_rate < 45:
        print("      ⚠️ Low win rate - review entry criteria")
    
    # R:R insight
    if rr >= 2.0:
        print("      ✅ Great R:R ratio - letting winners run")
    elif rr >= 1.5:
        print("      ✅ Decent R:R ratio")
    elif rr < 1.0:
        print("      ⚠️ R:R < 1 - winners smaller than losers")
        print("         Consider: wider TP or tighter SL")
    
    # Direction bias
    if longs and shorts:
        long_pnl = sum(t['pnl'] for t in longs)
        short_pnl = sum(t['pnl'] for t in shorts)
        if short_pnl > long_pnl * 2:
            print("      📊 SHORT trades significantly outperform")
        elif long_pnl > short_pnl * 2:
            print("      📊 LONG trades significantly outperform")
    
    # Duration insight
    if win_durs and loss_durs:
        avg_win_dur = sum(win_durs) / len(win_durs)
        avg_loss_dur = sum(loss_durs) / len(loss_durs)
        if avg_loss_dur > avg_win_dur * 1.5:
            print("      ⚠️ Holding losers too long - cut losses faster")
        if avg_win_dur < avg_loss_dur * 0.5:
            print("      ⚠️ Exiting winners too early - let them run")
    
    # Small wins big losses pattern
    if wins and losses:
        small_wins = len([t for t in wins if t['pnl'] < 0.1])
        big_losses = len([t for t in losses if t['pnl'] < -0.3])
        if small_wins > len(wins) * 0.5 and big_losses > len(losses) * 0.3:
            print("      ⚠️ PATTERN: Many small wins, few big losses")
            print("         This is a common retail trap - review SL placement")
    
    print()
    print("=" * 75)
    print()


def main():
    # Get address from env
    address = os.getenv('ACCOUNT_ADDRESS')
    testnet = os.getenv('TESTNET', 'false').lower() == 'true'
    symbol = os.getenv('SYMBOL', 'HYPE')
    
    if not address:
        print("❌ ACCOUNT_ADDRESS not set in .env")
        return
    
    print(f"🔍 Fetching fills from Hyperliquid API...")
    print(f"   Address: {address[:10]}...{address[-6:]}")
    print(f"   Network: {'Testnet' if testnet else 'Mainnet'}")
    
    fills = fetch_fills_from_api(address, testnet)
    print(f"   Found {len(fills)} total fills")
    
    # Filter for our symbol
    symbol_fills = [f for f in fills if f.get('coin') == symbol]
    print(f"   {len(symbol_fills)} fills for {symbol}")
    
    # Pair into trades
    print(f"\n🔄 Pairing fills into round-trip trades...")
    trades = pair_fills_into_trades(fills, symbol_filter=symbol)
    print(f"   Found {len(trades)} completed trades for {symbol}")
    
    # Analyze
    analyze_trades(trades, limit=100)


if __name__ == "__main__":
    main()
