#!/usr/bin/env python3
"""
Comprehensive Bot Debugging Script
Tests all components of the trading bot to find errors
"""

import asyncio
import sys
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    END = '\033[0m'

def success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def header(msg):
    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}{Colors.END}\n")

errors_found = []
warnings_found = []

# ============================================================
# STEP 1: Test All Imports
# ============================================================
def test_imports():
    header("STEP 1: Testing Module Imports")
    
    modules = [
        # Core
        ("app.bot", "HyperAIBot"),
        ("app.hl.hl_client", "HyperLiquidClient"),
        ("app.hl.hl_order_manager", "HLOrderManager"),
        ("app.hl.hl_websocket", "HLWebSocket"),
        
        # Database
        ("app.database.db_manager", "DatabaseManager"),
        ("app.database.analytics", "AnalyticsDashboard"),
        
        # Strategies
        ("app.strategies.strategy_manager", "StrategyManager"),
        ("app.strategies.rule_based.swing_strategy", "SwingStrategy"),
        
        # Risk
        ("app.risk.risk_engine", "RiskEngine"),
        ("app.risk.kill_switch", "KillSwitch"),
        ("app.risk.drawdown_monitor", "DrawdownMonitor"),
        
        # Portfolio
        ("app.portfolio.position_manager", "PositionManager"),
        ("app.portfolio.multi_asset_manager", "MultiAssetManager"),
        
        # Telegram
        ("app.tg_bot.bot", "TelegramBot"),
        ("app.tg_bot.formatters", "MessageFormatter"),
        ("app.tg_bot.keyboards", "KeyboardFactory"),
        
        # Utils
        ("app.utils.indicator_calculator", "IndicatorCalculator"),
        ("app.utils.symbol_manager", "SymbolManager"),
        ("app.utils.error_handler", "ErrorHandler"),
        
        # Adaptive strategies
        ("app.strategies.adaptive.market_regime", "MarketRegimeDetector"),
        ("app.strategies.adaptive.supertrend", "SupertrendIndicator"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls:
                success(f"{module_name}.{class_name}")
                passed += 1
            else:
                error(f"{module_name}.{class_name} - class not found")
                errors_found.append(f"Import: {module_name}.{class_name} not found")
                failed += 1
        except Exception as e:
            error(f"{module_name}.{class_name}")
            print(f"    Error: {e}")
            errors_found.append(f"Import: {module_name} - {e}")
            failed += 1
    
    print(f"\n  Passed: {passed}, Failed: {failed}")
    return failed == 0

# ============================================================
# STEP 2: Test Configuration
# ============================================================
def test_config():
    header("STEP 2: Testing Configuration (Environment Variables)")
    
    configs = [
        ("HYPERLIQUID_PRIVATE_KEY", 64, True),
        ("TELEGRAM_BOT_TOKEN", 40, True),
        ("TELEGRAM_CHAT_ID", 5, True),
        ("DATABASE_URL", 10, False),  # Optional
        ("HYPERLIQUID_TESTNET", 0, False),  # Optional
    ]
    
    all_ok = True
    for name, min_len, required in configs:
        value = os.getenv(name)
        if not value:
            if required:
                error(f"{name} is empty or not set")
                errors_found.append(f"Config: {name} is required but not set")
                all_ok = False
            else:
                warning(f"{name} is not set (optional)")
        elif len(str(value)) < min_len:
            warning(f"{name} seems too short: {len(str(value))} chars")
        else:
            if 'KEY' in name or 'TOKEN' in name or 'URL' in name:
                masked = str(value)[:8] + "..." + str(value)[-4:]
            else:
                masked = value
            success(f"{name}: {masked}")
    
    return all_ok

# ============================================================
# STEP 3: Test Database Connection
# ============================================================
async def test_database():
    header("STEP 3: Testing Database Connection")
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        warning("DATABASE_URL not set - skipping database test")
        info("Bot will use JSONL fallback for trade storage")
        return True
    
    try:
        from app.database.db_manager import DatabaseManager
        
        db = DatabaseManager(database_url)
        await db.initialize()
        success("Database connected and initialized")
        
        # Test a simple query
        result = await db.pool.fetchval("SELECT 1")
        if result == 1:
            success("Database query works")
        
        # Check tables exist
        tables = await db.pool.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_names = [t['table_name'] for t in tables]
        
        required_tables = ['trades', 'signals', 'positions', 'daily_stats']
        for table in required_tables:
            if table in table_names:
                success(f"Table '{table}' exists")
            else:
                warning(f"Table '{table}' not found - will be created on first run")
        
        await db.close()
        return True
        
    except Exception as e:
        error(f"Database error: {e}")
        errors_found.append(f"Database: {e}")
        traceback.print_exc()
        return False

# ============================================================
# STEP 4: Test Hyperliquid Connection
# ============================================================
async def test_hyperliquid():
    header("STEP 4: Testing Hyperliquid API")
    
    try:
        from app.hl.hl_client import HyperLiquidClient
        
        client = HyperLiquidClient()
        await client.initialize()
        success("Hyperliquid client initialized")
        
        # Test meta info
        meta = await client.get_meta()
        if meta and 'universe' in meta:
            success(f"Got market meta: {len(meta['universe'])} assets")
        else:
            warning("Meta info incomplete")
        
        # Test candle fetch
        candles = await client.get_candles("BTC", "15m", 50)
        if candles and len(candles) > 0:
            success(f"Fetched {len(candles)} candles for BTC")
            # Show latest price
            latest = candles[-1] if candles else None
            if latest:
                price = float(latest.get('c', 0))
                info(f"Latest BTC price: ${price:,.2f}")
        else:
            warning("No candles returned for BTC")
        
        # Test account info
        try:
            account = await client.get_account_state()
            if account:
                success("Got account state")
                # Show balance if available
                if 'marginSummary' in account:
                    balance = float(account['marginSummary'].get('accountValue', 0))
                    info(f"Account value: ${balance:,.2f}")
        except Exception as e:
            warning(f"Could not get account state: {e}")
        
        return True
        
    except Exception as e:
        error(f"Hyperliquid error: {e}")
        errors_found.append(f"Hyperliquid: {e}")
        traceback.print_exc()
        return False

# ============================================================
# STEP 5: Test Signal Generation
# ============================================================
async def test_signals():
    header("STEP 5: Testing Signal Generation")
    
    try:
        from app.strategies.rule_based.swing_strategy import SwingStrategy
        from app.hl.hl_client import HyperLiquidClient
        import pandas as pd
        
        client = HyperLiquidClient()
        await client.initialize()
        
        strategy = SwingStrategy()
        
        # Get candles for test
        candles = await client.get_candles("BTC", "15m", 200)
        
        if not candles or len(candles) < 100:
            warning("Not enough candles to test signals")
            return True
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df['open'] = df['o'].astype(float)
        df['high'] = df['h'].astype(float)
        df['low'] = df['l'].astype(float)
        df['close'] = df['c'].astype(float)
        df['volume'] = df['v'].astype(float)
        df = df.sort_values('timestamp')
        
        success(f"Prepared {len(df)} candles for signal test")
        
        # Test scoring directly
        try:
            long_score, short_score = strategy._calculate_scores(df)
            info(f"Raw scores - LONG: {long_score}/{strategy.max_signal_score}, SHORT: {short_score}/{strategy.max_signal_score}")
            info(f"Threshold: {strategy.min_signal_score}")
            
            if long_score < strategy.min_signal_score and short_score < strategy.min_signal_score:
                warning("Both scores below threshold - no signal will generate")
                warning(f"Consider lowering MIN_SIGNAL_SCORE from {strategy.min_signal_score}")
            elif long_score >= strategy.min_signal_score:
                success(f"LONG score ({long_score}) meets threshold!")
            elif short_score >= strategy.min_signal_score:
                success(f"SHORT score ({short_score}) meets threshold!")
        except Exception as e:
            warning(f"Could not calculate scores: {e}")
        
        # Test regime detection
        try:
            regime = strategy.market_regime_detector.detect(df)
            info(f"Current market regime: {regime.value}")
        except Exception as e:
            warning(f"Could not detect regime: {e}")
        
        return True
        
    except Exception as e:
        error(f"Signal generation error: {e}")
        errors_found.append(f"Signals: {e}")
        traceback.print_exc()
        return False

# ============================================================
# STEP 6: Test Telegram Bot
# ============================================================
async def test_telegram():
    header("STEP 6: Testing Telegram Bot")
    
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        warning("Telegram credentials not set - skipping test")
        return True
    
    try:
        from telegram import Bot
        
        bot = Bot(token=token)
        me = await bot.get_me()
        success(f"Connected to Telegram as @{me.username}")
        
        return True
        
    except Exception as e:
        error(f"Telegram error: {e}")
        errors_found.append(f"Telegram: {e}")
        traceback.print_exc()
        return False

# ============================================================
# STEP 7: Test Full Bot Initialization
# ============================================================
async def test_full_bot():
    header("STEP 7: Testing Full Bot Initialization")
    
    try:
        from app.bot import HyperAIBot
        
        bot = HyperAIBot()
        info("HyperAIBot instance created")
        
        # Initialize but don't run
        await bot.initialize()
        success("Bot initialized successfully")
        
        # Check components
        components = [
            ("hl_client", getattr(bot, 'hl_client', None)),
            ("strategy", getattr(bot, 'strategy', None)),
            ("risk_engine", getattr(bot, 'risk_engine', None)),
            ("kill_switch", getattr(bot, 'kill_switch', None)),
            ("position_manager", getattr(bot, 'position_manager', None)),
        ]
        
        for name, component in components:
            if component:
                success(f"Component '{name}' loaded")
            else:
                warning(f"Component '{name}' not found/initialized")
        
        # Clean up
        try:
            await bot.shutdown()
            success("Bot shutdown cleanly")
        except:
            pass
        
        return True
        
    except Exception as e:
        error(f"Full bot error: {e}")
        errors_found.append(f"Full bot: {e}")
        traceback.print_exc()
        return False

# ============================================================
# STEP 8: Check for Common Issues
# ============================================================
def check_common_issues():
    header("STEP 8: Checking Common Issues")
    
    # Check log directory
    log_dir = Path("logs")
    if log_dir.exists():
        success("Logs directory exists")
        log_files = list(log_dir.glob("*.log"))
        info(f"Found {len(log_files)} log files")
        
        # Check latest log for errors
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            with open(latest_log, 'r') as f:
                content = f.read()
                error_count = content.lower().count('error')
                exception_count = content.lower().count('exception')
                if error_count > 0 or exception_count > 0:
                    warning(f"Found {error_count} 'error' and {exception_count} 'exception' mentions in {latest_log.name}")
    else:
        warning("Logs directory doesn't exist - will be created")
    
    # Check data directories
    for dir_name in ["data/trades", "data/processed", "data/raw"]:
        if Path(dir_name).exists():
            success(f"Directory '{dir_name}' exists")
        else:
            warning(f"Directory '{dir_name}' missing - will be created")
    
    # Check .env file
    if Path(".env").exists():
        success(".env file exists")
    else:
        error(".env file missing - create from .env.example")
        errors_found.append("Missing .env file")
    
    return True

# ============================================================
# STEP 9: Show Configuration Summary
# ============================================================
def show_config_summary():
    header("STEP 9: Configuration Summary")
    
    from app.strategies.rule_based.swing_strategy import SwingStrategy
    
    strategy = SwingStrategy()
    
    print(f"{Colors.MAGENTA}Signal Generation Settings:{Colors.END}")
    print(f"  • Min Signal Score: {strategy.min_signal_score}/{strategy.max_signal_score} ({strategy.min_signal_score/strategy.max_signal_score:.0%})")
    print(f"  • Confirmation Scans: {os.getenv('SIGNAL_CONFIRMATION_SCANS', 3)}")
    print(f"  • Direction Lock: {os.getenv('DIRECTION_LOCK_SECONDS', 900)}s")
    print(f"  • Scan Interval: {os.getenv('SCAN_INTERVAL_SECONDS', 60)}s")
    
    print(f"\n{Colors.MAGENTA}Risk Management:{Colors.END}")
    print(f"  • Kill Switch Threshold: {os.getenv('KILL_SWITCH_THRESHOLD', -0.1)}")
    print(f"  • Max Drawdown: {os.getenv('MAX_DRAWDOWN_PCT', 0.05)}")
    print(f"  • Position Size: {os.getenv('POSITION_SIZE_PCT', 0.1)}")
    
    print(f"\n{Colors.MAGENTA}Trading Assets:{Colors.END}")
    assets = os.getenv('TRADING_ASSETS', 'BTC,ETH,SOL')
    print(f"  • Assets: {assets}")
    
    return True

# ============================================================
# MAIN
# ============================================================
async def main():
    print(f"\n{Colors.CYAN}{'='*60}")
    print("         HYPERBOT COMPREHENSIVE DEBUG")
    print(f"{'='*60}{Colors.END}")
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Database", await test_database()))
    results.append(("Hyperliquid", await test_hyperliquid()))
    results.append(("Signals", await test_signals()))
    results.append(("Telegram", await test_telegram()))
    results.append(("Full Bot", await test_full_bot()))
    results.append(("Common Issues", check_common_issues()))
    results.append(("Config Summary", show_config_summary()))
    
    # Summary
    header("FINAL SUMMARY")
    
    passed = 0
    failed = 0
    for name, result in results:
        if result:
            success(f"{name}: PASSED")
            passed += 1
        else:
            error(f"{name}: FAILED")
            failed += 1
    
    print(f"\n  Tests: {passed} passed, {failed} failed")
    
    if errors_found:
        print(f"\n{Colors.RED}{'='*60}")
        print("  ERRORS TO FIX:")
        print(f"{'='*60}{Colors.END}")
        for i, err in enumerate(errors_found, 1):
            print(f"  {i}. {err}")
    else:
        print(f"\n{Colors.GREEN}{'='*60}")
        print("  ALL TESTS PASSED - BOT IS READY!")
        print(f"{'='*60}{Colors.END}")
        print("\n  To start the bot:")
        print(f"  {Colors.CYAN}python -m app.bot{Colors.END}")
    
    return len(errors_found) == 0

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
