#!/usr/bin/env python3
"""Analyze trading performance from database"""
import asyncio
import asyncpg
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv('DATABASE_URL', 'postgresql://neondb_owner:npg_fIoF54ezKsTH@ep-fancy-grass-abs3vfwd-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require')

async def analyze_trades():
    conn = await asyncpg.connect(DB_URL)
    
    # Fetch last 100 closed trades
    trades = await conn.fetch("""
        SELECT 
            id, timestamp, symbol, signal_type, 
            entry_price, exit_price, quantity,
            pnl, pnl_percent, commission,
            duration_seconds, strategy_name, confidence_score,
            created_at, closed_at
        FROM trades 
        WHERE status = 'CLOSED'
        ORDER BY closed_at DESC 
        LIMIT 100
    """)
    
    if not trades:
        print("❌ No closed trades found in database")
        await conn.close()
        return
    
    print()
    print("="*70)
    print(f"📊 TRADE PERFORMANCE ANALYSIS - Last {len(trades)} Trades")
    print("="*70)
    print()
    
    # Basic stats
    total_pnl = sum(float(t['pnl'] or 0) for t in trades)
    wins = [t for t in trades if float(t['pnl'] or 0) > 0]
    losses = [t for t in trades if float(t['pnl'] or 0) < 0]
    breakeven = [t for t in trades if float(t['pnl'] or 0) == 0]
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    print("📈 OVERALL PERFORMANCE")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Total P&L: ${total_pnl:+.2f}")
    print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakeven)}BE)")
    
    if wins:
        avg_win = sum(float(t['pnl']) for t in wins) / len(wins)
        avg_win_pct = sum(float(t['pnl_percent'] or 0) for t in wins) / len(wins)
        best_win = max(float(t['pnl']) for t in wins)
        print()
        print("   ✅ Wins:")
        print(f"      Average Win: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
        print(f"      Best Win: ${best_win:.2f}")
    
    if losses:
        avg_loss = sum(float(t['pnl']) for t in losses) / len(losses)
        avg_loss_pct = sum(float(t['pnl_percent'] or 0) for t in losses) / len(losses)
        worst_loss = min(float(t['pnl']) for t in losses)
        print()
        print("   ❌ Losses:")
        print(f"      Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
        print(f"      Worst Loss: ${worst_loss:.2f}")
    
    # Risk/Reward analysis
    if wins and losses:
        avg_win_amt = sum(float(t['pnl']) for t in wins) / len(wins)
        avg_loss_amt = abs(sum(float(t['pnl']) for t in losses) / len(losses))
        rr_ratio = avg_win_amt / avg_loss_amt if avg_loss_amt > 0 else 0
        print()
        print(f"   📊 Risk/Reward Ratio: {rr_ratio:.2f}")
    
    # Profit Factor
    if losses:
        gross_profit = sum(float(t['pnl']) for t in wins) if wins else 0
        gross_loss = abs(sum(float(t['pnl']) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(f"   📊 Profit Factor: {profit_factor:.2f}")
    
    # Expectancy
    if trades:
        expectancy = total_pnl / len(trades)
        print(f"   📊 Expectancy per Trade: ${expectancy:.3f}")
    
    # By direction
    print()
    print("─"*70)
    print("📊 BY DIRECTION")
    
    longs = [t for t in trades if t['signal_type'] == 'BUY']
    shorts = [t for t in trades if t['signal_type'] == 'SELL']
    
    for direction, dtrades in [('LONG', longs), ('SHORT', shorts)]:
        if dtrades:
            d_pnl = sum(float(t['pnl'] or 0) for t in dtrades)
            d_wins = len([t for t in dtrades if float(t['pnl'] or 0) > 0])
            d_wr = d_wins / len(dtrades) * 100
            print(f"   {direction}: {len(dtrades)} trades | P&L: ${d_pnl:+.2f} | WR: {d_wr:.1f}%")
    
    # Duration analysis
    print()
    print("─"*70)
    print("⏱️ TRADE DURATION")
    
    durations = [t['duration_seconds'] for t in trades if t['duration_seconds']]
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
        win_durs = [t['duration_seconds'] for t in wins if t['duration_seconds']]
        loss_durs = [t['duration_seconds'] for t in losses if t['duration_seconds']]
        if win_durs:
            print(f"   Avg Win Duration: {format_dur(int(sum(win_durs)/len(win_durs)))}")
        if loss_durs:
            print(f"   Avg Loss Duration: {format_dur(int(sum(loss_durs)/len(loss_durs)))}")
    
    # P&L Distribution
    print()
    print("─"*70)
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
        count = len([t for t in trades if check(float(t['pnl'] or 0))])
        if count > 0:
            pct = count / len(trades) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {label:20} {bar} {count:3} ({pct:5.1f}%)")
    
    # Recent trades
    print()
    print("─"*70)
    print("📋 RECENT 15 TRADES")
    print("─"*70)
    
    for t in trades[:15]:
        pnl = float(t['pnl'] or 0)
        pnl_pct = float(t['pnl_percent'] or 0)
        emoji = '✅' if pnl > 0 else ('❌' if pnl < 0 else '⚪')
        direction = 'L' if t['signal_type'] == 'BUY' else 'S'
        closed = t['closed_at'].strftime('%m/%d %H:%M') if t['closed_at'] else 'N/A'
        dur = t['duration_seconds']
        dur_str = f"{dur//60}m" if dur and dur >= 60 else f"{dur}s" if dur else "?"
        print(f"   {emoji} {direction} {t['symbol']:6} | ${pnl:+7.4f} ({pnl_pct:+6.2f}%) | {dur_str:>6} | {closed}")
    
    # Time analysis
    print()
    print("─"*70)
    print("📅 DATE RANGE")
    
    if trades:
        oldest = min(t['created_at'] for t in trades if t['created_at'])
        newest = max(t['closed_at'] for t in trades if t['closed_at'])
        print(f"   From: {oldest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   To:   {newest.strftime('%Y-%m-%d %H:%M')}")
        
        days = (newest - oldest).days + 1
        print(f"   Span: {days} days")
        if days > 0:
            print(f"   Trades/Day: {len(trades)/days:.1f}")
            print(f"   P&L/Day: ${total_pnl/days:+.2f}")
    
    # Streak analysis
    print()
    print("─"*70)
    print("🔥 STREAKS")
    
    sorted_trades = sorted(trades, key=lambda t: t['closed_at'] or t['created_at'])
    
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0
    
    for t in sorted_trades:
        if float(t['pnl'] or 0) > 0:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        elif float(t['pnl'] or 0) < 0:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
        else:
            current_win = 0
            current_loss = 0
    
    print(f"   Max Win Streak: {max_win_streak}")
    print(f"   Max Loss Streak: {max_loss_streak}")
    
    # Current streak
    current_streak = 0
    streak_type = None
    for t in reversed(sorted_trades):
        pnl = float(t['pnl'] or 0)
        if streak_type is None:
            streak_type = 'win' if pnl > 0 else ('loss' if pnl < 0 else None)
            if streak_type:
                current_streak = 1
        elif streak_type == 'win' and pnl > 0:
            current_streak += 1
        elif streak_type == 'loss' and pnl < 0:
            current_streak += 1
        else:
            break
    
    if current_streak > 0:
        print(f"   Current Streak: {current_streak} {'wins 🔥' if streak_type == 'win' else 'losses 💀'}")
    
    # Drawdown analysis
    print()
    print("─"*70)
    print("📉 DRAWDOWN ANALYSIS")
    
    running_pnl = 0
    peak = 0
    max_dd = 0
    max_dd_pct = 0
    
    for t in sorted_trades:
        running_pnl += float(t['pnl'] or 0)
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd
    
    print(f"   Max Drawdown: ${max_dd:.2f}")
    print(f"   Peak P&L: ${peak:.2f}")
    print(f"   Final P&L: ${running_pnl:.2f}")
    
    # Summary verdict
    print()
    print("="*70)
    print("🎯 VERDICT")
    print("="*70)
    
    if profit_factor >= 1.5 and win_rate >= 50:
        print("   ✅ PROFITABLE - Good profit factor and win rate!")
    elif profit_factor >= 1.0:
        print("   ⚠️ MARGINAL - Breaking even, needs improvement")
    else:
        print("   ❌ LOSING - Strategy needs work")
    
    # Recommendations
    print()
    print("   💡 Insights:")
    if win_rate < 50:
        print("      - Low win rate - consider tighter entry criteria")
    if rr_ratio < 1.5:
        print("      - R:R ratio low - let winners run longer or cut losses faster")
    if len(losses) > 0 and abs(avg_loss) > avg_win * 1.5:
        print("      - Average loss > average win - tighten stop losses")
    if len(longs) > len(shorts) * 3:
        print("      - Heavy LONG bias - consider more SHORT opportunities")
    elif len(shorts) > len(longs) * 3:
        print("      - Heavy SHORT bias - consider more LONG opportunities")
    
    # Check if mostly small wins but big losses
    big_losses = len([t for t in losses if abs(float(t['pnl'])) > 0.3])
    small_wins = len([t for t in wins if float(t['pnl']) < 0.1])
    if big_losses > len(losses) * 0.3 and small_wins > len(wins) * 0.5:
        print("      - Pattern: Many small wins, few big losses - classic retail trap!")
    
    print()
    print("="*70)
    print()
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(analyze_trades())
