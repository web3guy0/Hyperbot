#!/usr/bin/env python3
"""
HyperAI Trader - Master Bot Controller
Orchestrates rule-based → AI mode transition with complete risk management

Phase 1: Rule-based scalping (collect 1,000-3,000 trades)
Phase 2: Train AI models on collected data
Phase 3: Hybrid AI + Rule validation
Phase 4: Full AI autonomy
"""

import asyncio
import signal
import sys
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, Optional, List
from types import FrameType
import json
import re

# Add to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs (tokens, API keys, URLs with tokens)"""
    
    def __init__(self):
        super().__init__()
        # Get sensitive tokens from environment
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.api_secret = os.getenv('HYPERLIQUID_API_SECRET', '')
        
    def filter(self, record):
        """Mask sensitive data in log records"""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            
            # Mask Telegram bot token in URLs
            if self.telegram_token and self.telegram_token in msg:
                # Mask the token
                if ':' in self.telegram_token:
                    bot_id = self.telegram_token.split(':')[0]
                    msg = msg.replace(self.telegram_token, f"{bot_id}:***MASKED***")
                else:
                    msg = msg.replace(self.telegram_token, "***MASKED***")
            
            # Mask API URLs with tokens using regex (catch any bot token in URLs)
            msg = re.sub(
                r'(https?://[^/]+/bot)(\d+:[A-Za-z0-9_-]+)',
                r'\1***MASKED***',
                msg
            )
            
            # Mask API secrets
            if self.api_secret and len(self.api_secret) > 10:
                msg = msg.replace(self.api_secret, '***MASKED***')
            
            record.msg = msg
            
        return True


# Import HyperLiquid integration
from app.hl.hl_client import HyperLiquidClient
from app.hl.hl_websocket import HLWebSocket
from app.hl.hl_order_manager import HLOrderManager

# Import strategies
from app.strategies.strategy_manager import StrategyManager

# Import Position Manager (manages manual orders + early exit)
from app.portfolio.position_manager import PositionManager

# Import Multi-Asset Manager
from app.portfolio.multi_asset_manager import MultiAssetManager, get_multi_asset_manager

# Import risk management (consolidated in app/)
from app.risk.risk_engine import RiskEngine
from app.risk.kill_switch import KillSwitch
from app.risk.drawdown_monitor import DrawdownMonitor
from app.risk.kelly_criterion import KellyCriterion, get_kelly_calculator
from app.risk.small_account_mode import SmallAccountMode, get_small_account_mode

# Import paper trading
from app.execution.paper_trading import PaperTradingEngine, is_paper_trading_enabled, get_paper_trading_balance

# Import Telegram bot (V2 with modern UX)
from app.tg_bot.bot import TelegramBot as TelegramBot

# Import error handler
from app.utils.error_handler import ErrorHandler

# Import database
from app.database.db_manager import DatabaseManager

# Phase 5: Shared indicator calculator
from app.utils.indicator_calculator import IndicatorCalculator

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Setup logging with sensitive data filter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/bot_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Add sensitive data filter to all handlers
sensitive_filter = SensitiveDataFilter()
for handler in logging.root.handlers:
    handler.addFilter(sensitive_filter)

logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from telegram library
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('telegram.ext').setLevel(logging.WARNING)

# Global shutdown event
shutdown_event = asyncio.Event()

def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle shutdown signals"""
    logger.info("\n🛑 Shutdown requested...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class AccountManagerProxy:
    """
    Dynamic proxy that always returns fresh values from the bot.
    
    Unlike static proxies that capture values at init time,
    this proxy uses properties to fetch current values on each access.
    This ensures risk_engine and kill_switch always see live data.
    """
    def __init__(self, bot: 'HyperAIBot'):
        self._bot = bot
    
    @property
    def current_equity(self):
        return self._bot.account_value
    
    @property
    def current_balance(self):
        return self._bot.account_value
    
    @property
    def peak_equity(self):
        return self._bot.peak_equity
    
    @property
    def session_start_equity(self):
        return self._bot.session_start_equity
    
    @property
    def session_pnl(self):
        return self._bot.session_pnl
    
    @property
    def margin_used(self):
        return self._bot.margin_used


class PositionManagerProxy:
    """Dynamic proxy for position manager."""
    def __init__(self, bot: 'HyperAIBot'):
        self._bot = bot
        self.open_positions = {}
    
    def get_position(self, symbol: str):
        return None


class HyperAIBot:
    """
    Master trading bot controller
    Manages strategy execution, risk controls, and mode transitions
    
    V3 Upgrade: Uses full SDK integration with:
    - Client Order IDs (cloid) for reliable order tracking
    - Atomic TPSL with grouping='normalTpsl'
    - Dead man's switch (schedule_cancel)
    - Real-time WebSocket subscriptions (userFills, orderUpdates, clearinghouseState)
    - Proper decimal rounding per SDK
    """
    
    def __init__(self):
        # Mode configuration
        self.mode = os.getenv('BOT_MODE', 'rule_based')  # rule_based, hybrid, ai
        # Trading symbol - can be any HyperLiquid asset (BTC, ETH, SOL, MATIC, etc.)
        self.symbol = os.getenv('SYMBOL', 'SOL')
        
        # TIMEFRAME: Configurable candle timeframe (1m, 5m, 15m)
        # Higher timeframes = fewer signals but higher quality
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        valid_timeframes = ['1m', '5m', '15m', '1h', '4h']
        if self.timeframe not in valid_timeframes:
            logger.warning(f"Invalid TIMEFRAME={self.timeframe}, using 1m")
            self.timeframe = '1m'
        
        # MULTI-ASSET TRADING: Trade multiple assets simultaneously
        self.multi_asset_mode = os.getenv('MULTI_ASSET_MODE', 'false').lower() == 'true'
        multi_assets_env = os.getenv('MULTI_ASSETS', 'BTC,ETH,SOL')
        self.multi_assets = [s.strip() for s in multi_assets_env.split(',') if s.strip()]
        self.max_positions = int(os.getenv('MAX_POSITIONS', '3'))
        
        # Multi-asset manager (initialized later if enabled)
        self.asset_manager: Optional[MultiAssetManager] = None
        
        # Strategies per symbol (for multi-asset mode)
        self.strategies: Dict[str, StrategyManager] = {}
        
        # Exchange components
        self.client: Optional[HyperLiquidClient] = None
        self.websocket: Optional[HLWebSocket] = None
        self.order_manager: Optional[HLOrderManager] = None
        
        # Strategy Manager (runs all 4 strategies) - single symbol mode
        self.strategy: Optional[StrategyManager] = None
        
        # Position Manager (manages manual orders + early exit)
        self.position_manager: Optional[PositionManager] = None
        
        # Risk management
        self.risk_engine: Optional[RiskEngine] = None
        self.kill_switch: Optional[KillSwitch] = None
        self.drawdown_monitor: Optional[DrawdownMonitor] = None
        
        # Kelly Criterion position sizing
        self.kelly: Optional[KellyCriterion] = None
        
        # Small Account Mode (auto-detected based on balance)
        self.small_account_mode: Optional[SmallAccountMode] = None
        
        # Paper Trading Mode (simulated trades for strategy validation)
        self.paper_trading: Optional[PaperTradingEngine] = None
        self.is_paper_trading = is_paper_trading_enabled()
        
        # Telegram bot
        self.telegram_bot: Optional[TelegramBot] = None
        
        # Position tracking with thread-safe lock
        self._position_details: Dict[str, Dict] = {}
        self._position_lock = asyncio.Lock()
        
        # Candle cache for strategies (reduces API calls by 98%)
        self._candles_cache: List[Dict[str, Any]] = []
        self._last_candle_fetch: Optional[datetime] = None
        self._candle_update_pending = False  # Track if we need fresh candles
        
        # BTC candles for correlation analysis (altcoins only)
        self._btc_candles_cache: List[Dict[str, Any]] = []
        self._last_btc_fetch: Optional[datetime] = None
        
        # HTF candles for multi-timeframe confirmation (MANDATORY for pro trading)
        self._htf_candles_cache: Dict[str, List[Dict[str, Any]]] = {}  # {interval: candles}
        self._last_htf_fetch: Optional[datetime] = None
        self._htf_intervals = ['15m', '1h', '4h']  # Always check these before entries
        
        # Phase 5: Shared indicator calculator (eliminates duplicate calculations)
        self.indicator_calc: Optional[IndicatorCalculator] = None
        
        # Phase 5 Part 2: Smart position monitoring (adaptive frequency)
        self._last_position_check: Optional[datetime] = None
        self._position_check_interval = 3.0  # Default 3 seconds
        self._atr_value: Optional[Decimal] = None  # Current ATR for adaptive monitoring
        
        # Trailing stop throttle - avoid spam updates (minimum 30 seconds between updates per symbol)
        self._last_trail_update: Dict[str, datetime] = {}
        self._trail_update_interval = 30  # Minimum seconds between trail updates
        
        # Track active trades for database closing
        self._active_trade_ids: Dict[str, Dict] = {}  # symbol -> {trade_id, entry_price, quantity, side, entry_time}
        
        # Error handler
        self.error_handler: Optional[ErrorHandler] = None
        
        # Database
        self.db: Optional[DatabaseManager] = None
        
        # Account tracking (simplified portfolio manager)
        self.account_value = Decimal('0')
        self.peak_equity = Decimal('0')
        self.session_start_equity = Decimal('0')
        self.session_pnl = Decimal('0')
        self.margin_used = Decimal('0')
        self.account_state: Dict[str, Any] = {}
        
        # State
        self.is_running = False
        self.is_paused = False  # For pause/resume functionality
        self.trades_executed = 0
        self.start_time: Optional[datetime] = None
        
        # Data collection for AI training
        self.trade_log_path = Path('data/trades')
        self.trade_log_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🤖 HyperAI Bot initialized")
        logger.info(f"   Mode: {self.mode}")
        logger.info(f"   Timeframe: {self.timeframe}")
        if self.multi_asset_mode:
            logger.info(f"   🌐 Multi-Asset Mode: ENABLED")
            logger.info(f"   Assets: {', '.join(self.multi_assets)}")
            logger.info(f"   Max Positions: {self.max_positions}")
        else:
            logger.info(f"   Symbol: {self.symbol}")
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("🔧 Initializing components...")
            
            # Check for Paper Trading Mode first
            if self.is_paper_trading:
                paper_balance = get_paper_trading_balance()
                self.paper_trading = PaperTradingEngine(paper_balance)
                logger.info("📝 ═══════════════════════════════════════")
                logger.info("📝 PAPER TRADING MODE ENABLED")
                logger.info(f"📝 Virtual Balance: ${paper_balance}")
                logger.info("📝 No real trades will be executed!")
                logger.info("📝 ═══════════════════════════════════════")
            
            # Load credentials
            account_address = os.getenv('ACCOUNT_ADDRESS')
            api_secret = os.getenv('API_SECRET')
            testnet = os.getenv('TESTNET', 'false').lower() == 'true'
            
            if not account_address or not api_secret:
                raise ValueError("Missing credentials in .env file")
            
            # Initialize exchange client
            self.client = HyperLiquidClient(
                account_address,
                api_secret,
                api_secret,
                testnet=testnet
            )
            
            # Initialize WebSocket with subscriptions
            self.websocket = HLWebSocket(
                address=account_address,
                testnet=testnet
            )
            
            # Link WebSocket to client for optimized get_account_state()
            self.client.websocket = self.websocket
            
            # Subscribe to user events and market data
            self.websocket.subscribe_all_mids()  # For price data
            self.websocket.subscribe_user_fills(self._on_fill)
            self.websocket.subscribe_order_updates(self._on_order_update)
            
            # Subscribe to candles based on mode
            if self.multi_asset_mode:
                # Multi-asset: subscribe to all assets
                for asset in self.multi_assets:
                    self.websocket.subscribe_candles(asset, self.timeframe, self._on_new_candle)
                logger.info(f"📊 Subscribed to {self.timeframe} candles for: {', '.join(self.multi_assets)}")
            else:
                # Single symbol mode
                self.websocket.subscribe_candles(self.symbol, self.timeframe, self._on_new_candle)
                logger.info(f"📊 Subscribed to {self.timeframe} candles for: {self.symbol}")
            
            self.websocket.start()
            logger.info("📊 WebSocket subscriptions active")
            
            # Initialize Order Manager
            self.order_manager = HLOrderManager(self.client)
            
            # Set leverage from MAX_LEVERAGE in .env
            leverage = int(os.getenv('MAX_LEVERAGE', '5'))
            
            # Initialize strategy manager(s) based on mode
            if self.multi_asset_mode:
                # MULTI-ASSET MODE: Create strategy and set leverage for each asset
                logger.info(f"🌐 Multi-Asset Mode: Initializing {len(self.multi_assets)} assets...")
                
                self.asset_manager = get_multi_asset_manager(
                    enabled_assets=self.multi_assets,
                    max_positions=self.max_positions
                )
                
                for asset in self.multi_assets:
                    self.order_manager.set_leverage(asset, leverage)
                    self.strategies[asset] = StrategyManager(asset)
                    logger.info(f"   ✅ {asset}: Strategy + {leverage}x leverage")
                
                # Use first asset's strategy as default for backward compatibility
                self.strategy = self.strategies.get(self.multi_assets[0])
                logger.info(f"🌐 Multi-Asset Mode: {len(self.strategies)} strategies initialized")
            else:
                # SINGLE SYMBOL MODE (original behavior)
                self.order_manager.set_leverage(self.symbol, leverage)
                logger.info(f"⚙️  Leverage set to {leverage}x for {self.symbol}")
                
                logger.info(f"🎯 Initializing Strategy Manager for {self.symbol}")
                self.strategy = StrategyManager(self.symbol)
            
            # Phase 5: Initialize shared indicator calculator
            self.indicator_calc = IndicatorCalculator()
            logger.info("📊 Phase 5: Shared indicator calculator initialized")
            
            # Phase 6: Initialize Position Manager (manages manual orders + early exit)
            position_manager_config = {
                'check_interval_seconds': 30,  # Check every 30 seconds
                'auto_tpsl': True,  # Auto-set TP/SL on unprotected positions
                'early_exit': True,  # Exit on failed setups
                'health_check': True,  # Monitor position health
                'trailing_stop': True,  # Enable trailing stops
                'break_even': True,  # Move SL to entry after profit
                'default_tp_pct': 3.0,  # Default TP = 3%
                'default_sl_pct': 1.5,  # Default SL = 1.5%
            }
            self.position_manager = PositionManager(
                client=self.client,
                order_manager=self.order_manager,
                strategy=self.strategy,
                config=position_manager_config
            )
            logger.info("🎯 Phase 6: Position Manager initialized (all features enabled)")
            
            # Get initial account state
            await self.update_account_state()
            
            # Check for small account mode (< $100)
            self.small_account_mode = get_small_account_mode(self.account_value)
            if self.small_account_mode and self.small_account_mode.is_small_account:
                logger.info("💰 SMALL ACCOUNT MODE ACTIVATED")
                logger.info(f"   Account value: ${self.account_value:.2f}")
                logger.info(f"   Tier: {self.small_account_mode.tier.upper()}")
                logger.info(f"   Recommended leverage: {self.small_account_mode.recommended_leverage}x")
                logger.info(f"   Tradeable assets: {', '.join(self.small_account_mode.get_tradeable_assets()[:3])}")
                
                # Apply small account config overrides
                self.small_account_mode.apply_config()
            
            # Initialize risk management components
            # Use dynamic proxies that always return fresh values
            account_manager_proxy = AccountManagerProxy(self)
            position_manager_proxy = PositionManagerProxy(self)
            
            risk_config = {
                'max_position_size_pct': float(os.getenv('MAX_POSITION_SIZE_PCT', '55')),
                'max_positions': int(os.getenv('MAX_POSITIONS', '1')),
                'max_leverage': int(os.getenv('MAX_LEVERAGE', '5')),
                'max_daily_loss_pct': float(os.getenv('MAX_DAILY_LOSS_PCT', '5')),
                'max_drawdown_pct': float(os.getenv('MAX_DRAWDOWN_PCT', '10'))
            }
            self.risk_engine = RiskEngine(account_manager_proxy, position_manager_proxy, risk_config)
            
            kill_switch_config = {
                'daily_loss_trigger_pct': float(os.getenv('KILL_SWITCH_DAILY_LOSS_PCT', os.getenv('MAX_DAILY_LOSS_PCT', '5'))),
                'drawdown_trigger_pct': float(os.getenv('KILL_SWITCH_DRAWDOWN_PCT', os.getenv('MAX_DRAWDOWN_PCT', '10')))
            }
            self.kill_switch = KillSwitch(account_manager_proxy, position_manager_proxy, kill_switch_config)
            
            drawdown_config = {
                'warning_threshold_pct': 5,
                'critical_threshold_pct': float(os.getenv('MAX_DRAWDOWN_PCT', '10')),
                'auto_pause_enabled': True,
                'auto_pause_threshold_pct': 12
            }
            self.drawdown_monitor = DrawdownMonitor(account_manager_proxy, drawdown_config)
            
            # Initialize Kelly Criterion position sizing
            kelly_enabled = os.getenv('KELLY_ENABLED', 'true').lower() == 'true'
            if kelly_enabled:
                kelly_fraction = float(os.getenv('KELLY_FRACTION', '0.5'))  # Half Kelly default
                kelly_min_trades = int(os.getenv('KELLY_MIN_TRADES', '20'))
                kelly_max_pct = float(os.getenv('KELLY_MAX_POSITION_PCT', '25'))
                self.kelly = KellyCriterion(
                    kelly_fraction=kelly_fraction,
                    min_trades=kelly_min_trades,
                    max_position_pct=kelly_max_pct
                )
                logger.info(f"📊 Kelly Criterion enabled: {kelly_fraction:.0%} Kelly, min {kelly_min_trades} trades")
            else:
                self.kelly = None
                logger.info("📊 Kelly Criterion disabled (using fixed position sizing)")
            
            # Initialize Telegram bot (if credentials provided)
            if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
                try:
                    logger.info("📱 Initializing Telegram bot...")
                    config = {
                        'max_leverage': int(os.getenv('MAX_LEVERAGE', '5')),
                        'max_daily_loss_pct': float(os.getenv('MAX_DAILY_LOSS_PCT', '5'))
                    }
                    self.telegram_bot = TelegramBot(self, config)
                    await self.telegram_bot.start()
                    
                    # Initialize error handler with Telegram
                    self.error_handler = ErrorHandler(self.telegram_bot)
                    logger.info("🛡️ Error handler initialized with Telegram notifications")
                    
                    # Initialize database if DATABASE_URL is set
                    database_url = os.getenv('DATABASE_URL')
                    if database_url:
                        try:
                            logger.info("📊 Connecting to PostgreSQL database...")
                            from app.database.db_manager import DatabaseManager
                            self.db = DatabaseManager(database_url)
                            await self.db.connect()
                            logger.info("✅ Database connected and migrations applied")
                        except Exception as db_error:
                            logger.error(f"❌ Database connection failed: {db_error}")
                            logger.warning("⚠️ Continuing without database (will use JSONL fallback)")
                            self.db = None
                    else:
                        logger.info("📊 DATABASE_URL not set, using JSONL fallback")
                    
                    # Start auto-trainer background task (if enabled)
                    auto_train_enabled = os.getenv('AUTO_TRAIN_ENABLED', 'true').lower() == 'true'
                    if auto_train_enabled:
                        logger.info("🤖 Starting ML auto-trainer...")
                        from ml.auto_trainer import AutoTrainer
                        self.auto_trainer = AutoTrainer()  # Uses env vars for config
                        asyncio.create_task(self.auto_trainer.schedule_daily_check(self.telegram_bot))
                    else:
                        logger.info("🤖 ML auto-trainer disabled (set AUTO_TRAIN_ENABLED=true to enable)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Telegram bot initialization failed: {e}")
                    self.telegram_bot = None
            else:
                logger.info("📱 Telegram bot disabled (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable)")
            
            logger.info("✅ All components initialized")
            logger.info(f"💰 Starting Balance: ${self.account_value:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    async def update_account_state(self):
        """Update account state from exchange"""
        try:
            account_state = await self.client.get_account_state()
            
            old_value = self.account_value
            self.account_value = Decimal(str(account_state['account_value']))
            self.margin_used = Decimal(str(account_state['margin_used']))
            
            # Only log if significant change (>$0.50) or first update
            value_change = abs(float(self.account_value - old_value)) if old_value > 0 else 999
            if value_change > 0.50 or old_value == 0:
                logger.info(f"📊 Account updated: value=${self.account_value:.2f}, margin=${self.margin_used:.2f}")
            
            # Update peak equity
            if self.account_value > self.peak_equity:
                self.peak_equity = self.account_value
            
            # Set session start on first update
            if self.session_start_equity == 0:
                self.session_start_equity = self.account_value
            
            # Calculate session P&L
            self.session_pnl = self.account_value - self.session_start_equity
            
        except Exception as e:
            logger.error(f"Error updating account state: {e}")
    
    def _on_new_candle(self, symbol: str, candle: Dict[str, Any]):
        """
        Callback for real-time candle updates (Phase 3 + Phase 4 + Phase 5)
        Triggers indicator recalculation on new candle
        Supports multi-asset mode
        """
        # Handle case where symbol or candle might be wrong type
        if isinstance(symbol, dict):
            # Data came in wrong order, try to extract
            candle = symbol
            symbol = candle.get('s', candle.get('coin', self.symbol)) if isinstance(candle, dict) else self.symbol
        if isinstance(candle, str):
            logger.debug(f"Received string candle: {candle[:50]}")
            return
            
        # Multi-asset mode: update the specific asset's cache
        if self.multi_asset_mode and self.asset_manager:
            if symbol in self.multi_assets:
                self.asset_manager.mark_candle_update_pending(symbol)
                
                # Invalidate strategy cache for this symbol
                if symbol in self.strategies:
                    strategy = self.strategies[symbol]
                    if hasattr(strategy, 'invalidate_indicator_cache'):
                        strategy.invalidate_indicator_cache()
                
                logger.debug(f"🕯️ New candle for {symbol} (multi-asset)")
        else:
            # Single symbol mode (original behavior)
            if symbol == self.symbol:
                self._candle_update_pending = True
                
                # PHASE 4: Invalidate strategy indicator cache on new candle
                if hasattr(self.strategy, 'invalidate_indicator_cache'):
                    self.strategy.invalidate_indicator_cache()
                
                # PHASE 5: Invalidate shared indicator calculator cache
                if self.indicator_calc:
                    self.indicator_calc.invalidate_cache()
                
                logger.debug(f"🕯️ New candle for {symbol} - invalidated all indicator caches")
    
    def _on_order_update(self, update: Dict[str, Any]):
        """
        Callback for real-time order updates (PHASE 4)
        Sends instant Telegram notifications when orders fill/cancel/trigger
        """
        try:
            # Handle case where update might be a string or have nested data
            if isinstance(update, str):
                logger.debug(f"Received string order update: {update[:100]}")
                return
            if not isinstance(update, dict):
                logger.debug(f"Received non-dict order update: {type(update)}")
                return
            
            # Extract data from nested structure if needed
            if 'data' in update:
                update = update.get('data', {})
                if isinstance(update, list) and len(update) > 0:
                    update = update[0]
            
            if not isinstance(update, dict):
                return
                
            status = update.get('status')
            order = update.get('order', {})
            if isinstance(order, str):
                order = {}
            
            coin = order.get('coin', 'UNKNOWN')
            side = order.get('side', 'unknown')
            size = order.get('sz', 0)
            price = order.get('limitPx') or order.get('triggerPx', 0)
            
            # Send Telegram notification for important events
            if self.telegram_bot and status in ['filled', 'triggered', 'canceled']:
                message = None
                
                if status == 'filled':
                    message = f"✅ **ORDER FILLED**\n\n{coin} {side.upper()} {size} @ ${price}"
                elif status == 'triggered':
                    # Stop loss or take profit triggered
                    order_type = order.get('orderType', 'stop')
                    if 'tp' in order_type.lower():
                        message = f"🎯 **TAKE PROFIT HIT**\n\n{coin} closed @ ${price}\nProfit secured! 💰"
                    else:
                        message = f"🛑 **STOP LOSS HIT**\n\n{coin} closed @ ${price}\nLoss limited, capital protected."
                elif status == 'canceled':
                    message = f"🚫 **ORDER CANCELLED**\n\n{coin} {side.upper()} {size} @ ${price}"
                
                if message:
                    # Send async notification (don't block)
                    asyncio.create_task(self._send_order_notification(message))
                    
        except Exception as e:
            logger.error(f"Error in order update callback: {e}")
    
    async def _send_order_notification(self, message: str):
        """Send Telegram notification asynchronously"""
        try:
            if self.telegram_bot:
                await self.telegram_bot.send_message(message)
        except Exception as e:
            logger.debug(f"Telegram notification failed: {e}")
    
    def _on_fill(self, fill: Dict[str, Any]):
        """
        Callback for real-time fill updates
        Called when orders are filled on the exchange
        """
        try:
            # Handle case where fill might be a string or have nested data
            if isinstance(fill, str):
                logger.debug(f"Received string fill: {fill[:100]}")
                return
            if not isinstance(fill, dict):
                logger.debug(f"Received non-dict fill: {type(fill)}")
                return
            
            # Extract data from nested structure if needed
            if 'data' in fill:
                fill = fill.get('data', {})
                if isinstance(fill, list) and len(fill) > 0:
                    fill = fill[0]
            
            if not isinstance(fill, dict):
                return
                
            coin = fill.get('coin', 'UNKNOWN')
            side = fill.get('side', 'unknown')
            size = fill.get('sz', 0)
            price = fill.get('px', 0)
            fee = fill.get('fee', 0)
            closed_pnl = fill.get('closedPnl', 0)
            cloid = fill.get('cloid')
            
            # Format log message
            pnl_str = f" | P&L: ${closed_pnl:+.2f}" if closed_pnl else ""
            cloid_str = f" [cloid: {cloid}]" if cloid else ""
            
            logger.info(f"📥 FILL: {coin} {side.upper()} {size} @ ${price} | Fee: ${fee}{pnl_str}{cloid_str}")
            
            # Track fill in order manager if using cloid
            if cloid and hasattr(self.order_manager, 'track_fill'):
                self.order_manager.track_fill(cloid, fill)
            
            # Schedule Telegram notification for fills (handle no event loop gracefully)
            if self.telegram_bot and closed_pnl:  # Only notify on closes with P&L
                try:
                    emoji = "🟢" if side == "B" else "🔴"
                    pnl_emoji = "✅" if closed_pnl > 0 else "❌"
                    message = f"{emoji} **FILL**\n\n{coin} {side.upper()} {size} @ ${price}\n{pnl_emoji} Closed P&L: ${closed_pnl:+.2f}"
                    
                    # Try to get running loop, if not available just skip notification
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._send_order_notification(message))
                    except RuntimeError:
                        # No running event loop - skip async notification
                        pass
                except Exception:
                    pass
                
        except Exception as e:
            logger.error(f"Error in fill callback: {e}")
    
    async def _monitor_positions(self, account_state: Dict[str, Any]):
        """
        PHASE 5: Smart position monitoring with adaptive frequency
        
        Monitors active positions with volatility-based check frequency:
        - High volatility (ATR > 0.5%): Check every 2s
        - Medium volatility (ATR 0.2-0.5%): Check every 3s
        - Low volatility (ATR < 0.2%): Check every 5s
        
        Provides multi-layer safety:
        1. Position existence tracking
        2. Unrealized P&L vs SL/TP levels
        3. Dynamic trailing stops
        4. Position size change detection
        """
        try:
            # PHASE 5: Adaptive monitoring frequency based on volatility
            now = datetime.now(timezone.utc)
            
            # Calculate adaptive check interval based on ATR (volatility)
            if self._atr_value is not None:
                atr_pct = self._atr_value
                if atr_pct > Decimal('0.5'):
                    # High volatility: Monitor aggressively (every 2s)
                    self._position_check_interval = 2.0
                elif atr_pct > Decimal('0.2'):
                    # Medium volatility: Normal monitoring (every 3s)
                    self._position_check_interval = 3.0
                else:
                    # Low volatility: Relaxed monitoring (every 5s)
                    self._position_check_interval = 5.0
            
            # Skip check if not enough time elapsed (adaptive throttling)
            if self._last_position_check is not None:
                time_since_check = (now - self._last_position_check).total_seconds()
                if time_since_check < self._position_check_interval:
                    return  # Skip this check - not time yet
            
            # Update last check time
            self._last_position_check = now
            
            positions = account_state.get('positions', [])
            current_symbols = {pos['symbol'] for pos in positions if float(pos.get('size', 0)) != 0}
            
            # Update position details with lock to prevent race conditions
            async with self._position_lock:
                # Update position details for active positions
                for pos in positions:
                    size = float(pos.get('size', 0))
                    if size != 0:
                        symbol = pos['symbol']
                        self._position_details[symbol] = {
                            'entry_price': float(pos.get('entry_price', 0)),
                            'size': abs(size),
                            'side': 'long' if size > 0 else 'short',
                            'unrealized_pnl': float(pos.get('unrealized_pnl', 0))
                        }
                
                # Track symbols that closed since last check
                if hasattr(self, '_last_positions'):
                    closed_positions = self._last_positions - current_symbols
                    for symbol in closed_positions:
                        logger.info(f"🔄 Position closed: {symbol}")
                        
                        # Close trade in database if we have a trade_id
                        if self.db and symbol in self._active_trade_ids:
                            trade_info = self._active_trade_ids[symbol]
                            details = self._position_details.get(symbol, {})
                            realized_pnl = details.get('unrealized_pnl', 0)  # Now realized
                            entry_price = trade_info.get('entry_price', details.get('entry_price', 0))
                            size = trade_info.get('quantity', details.get('size', 0))
                            
                            # Calculate exit price and PnL percent
                            if size > 0 and entry_price > 0:
                                side = trade_info.get('side', details.get('side', 'long'))
                                if side.lower() in ('buy', 'long'):
                                    exit_price = entry_price + (realized_pnl / size)
                                else:
                                    exit_price = entry_price - (realized_pnl / size)
                                pnl_percent = (realized_pnl / (size * entry_price)) * 100
                            else:
                                exit_price = entry_price
                                pnl_percent = 0
                            
                            # Calculate duration
                            entry_time = trade_info.get('entry_time')
                            duration_seconds = int((datetime.now(timezone.utc) - entry_time).total_seconds()) if entry_time else None
                            
                            try:
                                await self.db.close_trade(
                                    trade_id=trade_info['trade_id'],
                                    exit_price=exit_price,
                                    pnl=realized_pnl,
                                    pnl_percent=pnl_percent,
                                    commission=0.0,  # TODO: Calculate actual commission
                                    duration_seconds=duration_seconds
                                )
                                logger.info(f"📊 DB: Trade #{trade_info['trade_id']} closed: P&L ${realized_pnl:+.2f} ({pnl_percent:+.2f}%)")
                            except Exception as db_err:
                                logger.error(f"❌ Failed to close trade in DB: {db_err}")
                            finally:
                                del self._active_trade_ids[symbol]
                        
                        # Record trade to Kelly Criterion if we have details
                        if self.kelly and symbol in self._position_details:
                            details = self._position_details[symbol]
                            realized_pnl = details.get('unrealized_pnl', 0)  # Now realized
                            entry_price = details['entry_price']
                            size = details['size']
                            side = details['side']
                            
                            # Calculate exit_price from PnL: PnL = (exit - entry) * size for long
                            # For long: exit = entry + PnL/size; For short: exit = entry - PnL/size
                            if size > 0 and entry_price > 0:
                                if side == 'long':
                                    exit_price = entry_price + (realized_pnl / size)
                                else:
                                    exit_price = entry_price - (realized_pnl / size)
                            else:
                                exit_price = entry_price  # Fallback
                            
                            self.kelly.add_trade(
                                pnl=realized_pnl,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                size=size,
                                side=side
                            )
                            logger.info(f"📊 Kelly: Recorded trade {symbol} P&L: ${realized_pnl:+.2f}")
                            del self._position_details[symbol]
                        
                        # Clean up backup targets
                        if hasattr(self.order_manager, 'position_targets') and symbol in self.order_manager.position_targets:
                            del self.order_manager.position_targets[symbol]
            
            # Monitor each active position
            for pos in positions:
                size = float(pos.get('size', 0))
                if size == 0:
                    continue
                
                symbol = pos['symbol']
                entry_price = Decimal(str(pos.get('entry_price', 0)))
                unrealized_pnl = Decimal(str(pos.get('unrealized_pnl', 0)))
                # Prevent division by zero if entry_price=0
                if entry_price > 0:
                    unrealized_pnl_pct = (unrealized_pnl / (abs(Decimal(str(size))) * entry_price)) * 100
                else:
                    unrealized_pnl_pct = Decimal('0')
                current_price = Decimal(str(pos.get('mark_price', entry_price)))  # Use mark price
                
                # Log position status every 60 loops (~1 min) for cleaner logs
                if hasattr(self, '_position_log_counter'):
                    self._position_log_counter += 1
                else:
                    self._position_log_counter = 1
                
                if self._position_log_counter % 60 == 0:
                    logger.info(f"📈 Active position: {symbol} | Size: {size:.2f} | "
                               f"Entry: ${entry_price:.3f} | Current: ${current_price:.3f} | P&L: ${unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.1f}%)")
                
                # DYNAMIC TRAILING STOPS - Lock in profits as they grow!
                # Uses OrderManagerV2 position tracking (no need for position_targets)
                if symbol in self.order_manager.position_orders:
                    order_info = self.order_manager.position_orders[symbol]
                    is_long = size > 0
                    
                    # Store entry price in order manager for trailing calculation
                    self.order_manager.set_entry_price(symbol, float(entry_price))
                    
                    # ==================== TRAILING STOP THROTTLE ====================
                    # Avoid spam orders - only update trailing stops every 30 seconds per symbol
                    now = datetime.now(timezone.utc)
                    last_trail = self._last_trail_update.get(symbol)
                    can_update_trail = (
                        last_trail is None or 
                        (now - last_trail).total_seconds() >= self._trail_update_interval
                    )
                    
                    # ==================== SWING TRAILING ====================
                    # At +7% PnL: Move SL to lock in +3% profit (breakeven + buffer)
                    # TP stays UNTOUCHED - let the original TP hit
                    
                    # Only trail SL, never touch TP
                    if unrealized_pnl_pct >= 7.0 and can_update_trail:
                        # Calculate new SL at +3% PnL (0.6% price with 5x leverage)
                        if is_long:
                            # Move SL to entry + 0.6% price = +3% PnL locked
                            trailing_sl = float(entry_price * Decimal('1.006'))
                            if trailing_sl > float(current_price):
                                trailing_sl = float(entry_price * Decimal('1.003'))  # Fallback to +1.5% PnL
                        else:
                            # Short: Move SL to entry - 0.6% price = +3% PnL locked
                            trailing_sl = float(entry_price * Decimal('0.994'))
                            if trailing_sl < float(current_price):
                                trailing_sl = float(entry_price * Decimal('0.997'))  # Fallback to +1.5% PnL
                        
                        # Only update if we haven't already trailed to this level
                        current_sl = order_info.get('sl_price', 0)
                        sl_improved = (is_long and trailing_sl > current_sl) or (not is_long and trailing_sl < current_sl)
                        
                        if sl_improved or current_sl == 0:
                            logger.info(f"🔒 TRAILING SL: At {unrealized_pnl_pct:.1f}% PnL - Moving SL to ${trailing_sl:.4f} (locks +3% PnL)")
                            
                            try:
                                # ONLY modify SL, keep TP untouched!
                                result = await self.order_manager.modify_stops(
                                    symbol=symbol,
                                    new_sl=trailing_sl,
                                    new_tp=None  # Explicitly None = don't touch TP
                                )
                                if result.get('success'):
                                    self._last_trail_update[symbol] = now
                                    order_info['sl_price'] = trailing_sl
                                    logger.info(f"✅ SL updated on exchange (TP unchanged)")
                                else:
                                    logger.warning(f"⚠️ Failed to update SL: {result.get('error')}")
                            except Exception as e:
                                logger.error(f"❌ Error updating trailing SL: {e}")
                
                # Check if approaching SL/TP levels (warning system)
                if unrealized_pnl_pct <= -0.8:  # Within 80% of typical -1% SL
                    logger.warning(f"⚠️  {symbol} approaching stop loss: P&L {unrealized_pnl_pct:+.1f}%")
                elif unrealized_pnl_pct >= 1.6:  # Within 80% of typical +2% TP
                    logger.info(f"🎯 {symbol} approaching take profit: P&L {unrealized_pnl_pct:+.1f}%")
            
            # Store current positions for next iteration
            self._last_positions = current_symbols
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}", exc_info=True)
    
    async def _scan_multi_asset_signals(self, account_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Scan all enabled assets for trading signals (multi-asset mode)
        
        Uses round-robin scanning to give each asset fair opportunity.
        Only scans assets without open positions.
        
        Returns:
            Signal dict if found, None otherwise
        """
        if not self.asset_manager:
            return None
        
        # Update asset manager with current positions
        self.asset_manager.update_from_account_state(account_state)
        
        # Check if we can open more positions
        if not self.asset_manager.can_open_new_position():
            return None
        
        # Get assets available for trading
        available = self.asset_manager.get_assets_without_positions()
        if not available:
            return None
        
        now = datetime.now(timezone.utc)
        
        # Scan each available asset
        for symbol in available:
            can_trade, reason = self.asset_manager.can_trade_asset(symbol)
            if not can_trade:
                logger.debug(f"⏭️ Skip {symbol}: {reason}")
                continue
            
            # Get strategy for this symbol
            strategy = self.strategies.get(symbol)
            if not strategy:
                logger.warning(f"No strategy for {symbol}")
                continue
            
            # Get market data
            market_data = self.websocket.get_market_data(symbol)
            if not market_data or not market_data.get('price'):
                continue
            
            # Fetch candles if needed (using configured timeframe)
            if self.asset_manager.needs_candle_refresh(symbol):
                candles = self.client.get_candles(symbol, self.timeframe, 150)
                if candles:
                    self.asset_manager.update_candles(symbol, candles)
            
            # Get cached candles
            candles = self.asset_manager.get_candles(symbol)
            if not candles:
                continue
            
            market_data['candles'] = candles
            
            # Update BTC candles for correlation (non-BTC assets)
            if symbol != 'BTC' and hasattr(strategy, 'update_btc_candles'):
                if self._btc_candles_cache:
                    strategy.update_btc_candles(self._btc_candles_cache)
            
            # Generate signal
            try:
                signal = await strategy.generate_signal(market_data, account_state)
                
                if signal:
                    # Validate signal is a dict
                    if not isinstance(signal, dict):
                        logger.warning(f"⚠️ {symbol} signal is {type(signal)}, not dict - skipping")
                        continue
                    
                    # Record signal time for cooldown
                    self.asset_manager.record_signal(symbol)
                    logger.info(f"🌐 Multi-Asset Signal: {symbol} {signal.get('signal_type', 'UNKNOWN')}")
                    return signal
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return None

    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting trading loop...")
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        loop_count = 0
        
        # IMMEDIATE STARTUP: Detect existing positions and protect them
        if self.position_manager:
            logger.info("🔍 Scanning for existing positions on startup...")
            try:
                existing_positions = await self.position_manager.scan_positions()
                if existing_positions:
                    logger.info(f"📍 Found {len(existing_positions)} existing position(s):")
                    for pos in existing_positions:
                        logger.info(f"   • {pos.symbol} {pos.side.upper()} @ ${pos.entry_price:.4f} "
                                   f"(PnL: {pos.unrealized_pnl_pct:+.2f}%)")
                        # Immediately try to set TP/SL
                        await self.position_manager.manage_position(pos, self._candles_cache)
                else:
                    logger.info("   No existing positions found")
            except Exception as e:
                logger.error(f"Error scanning positions on startup: {e}")
        
        # Start Position Manager monitoring loop (runs in parallel)
        position_manager_task = None
        if self.position_manager:
            logger.info("🔄 Starting Position Manager monitoring loop...")
            position_manager_task = asyncio.create_task(
                self._run_position_manager_loop()
            )
        
        try:
            while not shutdown_event.is_set() and self.is_running:
                loop_count += 1
                
                try:
                    # Check if bot is paused
                    if self.is_paused:
                        await asyncio.sleep(5)
                        continue
                    
                    # Check kill switch
                    if self.kill_switch.check_triggers():
                        logger.critical("🚨 KILL SWITCH ACTIVATED")
                        if self.telegram_bot:
                            try:
                                await self.telegram_bot.notify_emergency("Kill switch activated! Trading stopped.")
                            except Exception:
                                pass
                        break
                    
                    # Update account state every 10 loops
                    if loop_count % 10 == 0:
                        await self.update_account_state()
                        # Note: No manual proxy update needed - AccountManagerProxy uses @property
                        # to always return fresh values from self.account_value, etc.
                    
                    # Check drawdown
                    self.drawdown_monitor.update()
                    if self.drawdown_monitor.is_paused:
                        logger.warning("⏸️  Paused due to drawdown")
                        await asyncio.sleep(30)
                        continue
                    
                    # Get market data
                    market_data = self.websocket.get_market_data(self.symbol)
                    
                    if not market_data or not market_data.get('price'):
                        await asyncio.sleep(1)
                        continue
                    
                    # PHASE 3 OPTIMIZATION: Use WebSocket candles + smart fallback
                    # Only fetch via API on first run or if WebSocket fails
                    now = datetime.now(timezone.utc)
                    need_initial_fetch = not self._candles_cache and not self._last_candle_fetch
                    need_fallback_fetch = (
                        self._last_candle_fetch and 
                        (now - self._last_candle_fetch).total_seconds() > 900  # 15 min fallback
                    )
                    
                    if need_initial_fetch or need_fallback_fetch:
                        # Initial fetch or fallback if WebSocket not providing candles
                        candles = self.client.get_candles(self.symbol, self.timeframe, 150)
                        if candles:
                            self._candles_cache = candles
                            self._last_candle_fetch = now
                            logger.debug(f"📊 Fetched {self.timeframe} candles via API: {len(candles)} bars (fallback)")
                    elif self._candle_update_pending:
                        # WebSocket provided new candle - just refresh the cache
                        candles = self.client.get_candles(self.symbol, self.timeframe, 150)
                        if candles:
                            self._candles_cache = candles
                            self._last_candle_fetch = now
                            self._candle_update_pending = False
                            logger.debug(f"📊 Updated {self.timeframe} candles on new bar: {len(candles)} bars")
                    
                    # WORLD-CLASS: Fetch BTC candles for correlation analysis (altcoins only)
                    if self.symbol != 'BTC' and hasattr(self.strategy, 'update_btc_candles'):
                        btc_need_fetch = (
                            not self._btc_candles_cache or 
                            not self._last_btc_fetch or
                            (now - self._last_btc_fetch).total_seconds() > 300  # 5 min refresh
                        )
                        if btc_need_fetch:
                            try:
                                btc_candles = self.client.get_candles('BTC', self.timeframe, 50)
                                if btc_candles:
                                    self._btc_candles_cache = btc_candles
                                    self._last_btc_fetch = now
                                    self.strategy.update_btc_candles(btc_candles)
                                    logger.debug(f"₿ Updated BTC {self.timeframe} candles: {len(btc_candles)} bars")
                            except Exception as e:
                                logger.debug(f"Failed to fetch BTC candles: {e}")
                    
                    # PRO TRADING: Fetch HTF candles for multi-timeframe confirmation (MANDATORY)
                    # Only fetch if LTF is short-term (1m, 5m) - no need if already on 1h/4h
                    if self.timeframe in ['1m', '5m', '15m']:
                        htf_need_fetch = (
                            not self._htf_candles_cache or 
                            not self._last_htf_fetch or
                            (now - self._last_htf_fetch).total_seconds() > 900  # 15 min refresh
                        )
                        if htf_need_fetch:
                            try:
                                for interval in self._htf_intervals:
                                    # Skip if HTF equals our LTF
                                    if interval == self.timeframe:
                                        continue
                                    htf_candles = self.client.get_candles(self.symbol, interval, 50)
                                    if htf_candles:
                                        self._htf_candles_cache[interval] = htf_candles
                                self._last_htf_fetch = now
                                logger.debug(f"📊 Updated HTF candles: {list(self._htf_candles_cache.keys())}")
                            except Exception as e:
                                logger.debug(f"Failed to fetch HTF candles: {e}")
                    
                    # Always use cached candles for strategies
                    if self._candles_cache:
                        market_data['candles'] = self._candles_cache
                    
                    # Add HTF candles to market_data for multi-timeframe analysis (MANDATORY)
                    if self._htf_candles_cache:
                        market_data['htf_candles'] = self._htf_candles_cache
                    
                    # Add BTC candles to market_data for correlation analysis
                    if self._btc_candles_cache:
                        market_data['btc_candles'] = self._btc_candles_cache
                    
                    # Update account state
                    await self.update_account_state()
                    
                    # Get account state for strategy
                    account_state = await self.client.get_account_state()
                    
                    # Monitor active positions for SL/TP hits
                    await self._monitor_positions(account_state)
                    
                    # PAPER TRADING: Update paper positions with current prices
                    if self.is_paper_trading and self.paper_trading:
                        current_prices = {}
                        for symbol in self.paper_trading.positions.keys():
                            md = self.websocket.get_market_data(symbol)
                            if md and md.get('price'):
                                current_prices[symbol] = Decimal(str(md['price']))
                        
                        closed_trades = self.paper_trading.update_positions(current_prices)
                        for trade in closed_trades:
                            emoji = "🟢" if trade.pnl > 0 else "🔴"
                            logger.info(f"📝 [PAPER] {emoji} Position closed: {trade.symbol} P&L: ${trade.pnl:.2f}")
                            if self.telegram_bot:
                                try:
                                    await self.telegram_bot.send_message(
                                        f"📝 **PAPER TRADE CLOSED**\n"
                                        f"{emoji} {trade.side.upper()} {trade.symbol}\n"
                                        f"P&L: ${float(trade.pnl):.2f} ({float(trade.pnl_pct):+.2f}%)\n"
                                        f"Reason: {trade.exit_reason}"
                                    )
                                except Exception:
                                    pass
                    
                    # Skip signal generation if paused
                    if self.is_paused:
                        await asyncio.sleep(2)
                        continue
                    
                    # ========== SIGNAL GENERATION (Single vs Multi-Asset Mode) ==========
                    signal = None
                    active_strategy = self.strategy  # Default strategy
                    
                    if self.multi_asset_mode:
                        # MULTI-ASSET MODE: Scan all enabled assets for signals
                        signal = await self._scan_multi_asset_signals(account_state)
                        
                        if signal:
                            # Get the strategy that generated this signal
                            signal_symbol = signal.get('symbol', self.symbol)
                            active_strategy = self.strategies.get(signal_symbol, self.strategy)
                    else:
                        # SINGLE SYMBOL MODE (original behavior)
                        # Skip if position already open (max 1 position in single mode)
                        positions = account_state.get('positions', [])
                        has_open_position = any(float(pos.get('size', 0)) != 0 for pos in positions)
                        
                        if has_open_position:
                            # Don't generate new signals while position is active
                            await asyncio.sleep(1)
                            continue
                        
                        # PHASE 5: Calculate indicators once using shared calculator
                        if self._candles_cache and self.indicator_calc:
                            # Extract prices from candles
                            prices_list = [Decimal(str(c['close'])) for c in self._candles_cache]
                            volumes_list = [Decimal(str(c['volume'])) for c in self._candles_cache]
                            
                            # Calculate all indicators once (pass candles for proper ADX/ATR)
                            shared_indicators = self.indicator_calc.calculate_all(
                                prices_list, volumes_list, candles=self._candles_cache
                            )
                            
                            # Add to market_data
                            market_data['indicators'] = shared_indicators
                            
                            # PHASE 5 Part 2: Extract ATR for adaptive position monitoring
                            self._atr_value = shared_indicators.get('atr')
                        
                        # Generate signal from strategy (single symbol mode)
                        signal = await self.strategy.generate_signal(market_data, account_state)
                    
                    # DEFENSIVE: Ensure signal is a valid dict before processing
                    if signal:
                        if isinstance(signal, str):
                            logger.warning(f"⚠️ Signal is string, not dict: {signal[:100]}")
                            signal = None
                        elif not isinstance(signal, dict):
                            logger.warning(f"⚠️ Signal is {type(signal)}, not dict")
                            signal = None
                    
                    if signal:
                        # ==================== RECOVERY MODE CHECK ====================
                        # Reduce position sizes after drawdown to limit further losses
                        if self.drawdown_monitor:
                            trading_allowed, dd_reason = self.drawdown_monitor.is_trading_allowed()
                            if not trading_allowed:
                                logger.warning(f"⛔ Signal rejected: {dd_reason}")
                                await asyncio.sleep(5)
                                continue
                            
                            recovery_multiplier = self.drawdown_monitor.get_recovery_mode_multiplier()
                            if recovery_multiplier < 1.0:
                                original_size = signal['size']
                                signal['size'] = float(Decimal(str(original_size)) * Decimal(str(recovery_multiplier)))
                                logger.info(f"🔄 Recovery mode: size {original_size:.4f} → {signal['size']:.4f} ({recovery_multiplier:.0%})")
                        
                        # Apply Kelly Criterion position sizing adjustment
                        if self.kelly:
                            kelly_result = self.kelly.calculate()
                            if kelly_result.confidence > 0.3:  # Only adjust if we have confidence
                                # Adjust position size based on Kelly
                                original_size = signal['size']
                                kelly_adjusted_pct = min(
                                    kelly_result.position_size_pct,
                                    float(os.getenv('MAX_POSITION_SIZE_PCT', '55'))
                                )
                                # Scale the token size proportionally
                                if signal.get('position_size_pct', 50) > 0:
                                    size_ratio = kelly_adjusted_pct / signal.get('position_size_pct', 50)
                                    signal['size'] = float(Decimal(str(original_size)) * Decimal(str(size_ratio)))
                                    signal['position_size_pct'] = kelly_adjusted_pct
                                    logger.info(f"📊 Kelly adjusted size: {original_size:.4f} → {signal['size']:.4f} ({kelly_adjusted_pct:.1f}%)")
                        
                        # Send Telegram notification for new signal
                        if self.telegram_bot:
                            try:
                                await self.telegram_bot.notify_signal(signal)
                            except Exception:
                                pass  # Don't let Telegram errors stop trading
                        
                        # Validate with risk engine
                        is_valid, rejection_reason = self.risk_engine.validate_pre_trade(
                            signal['symbol'],
                            signal['side'],
                            Decimal(str(signal['size'])),
                            Decimal(str(signal['entry_price']))
                        )
                        
                        if is_valid:
                            # **REVALIDATE SIGNAL** - Check if still valid before execution
                            # Get market data for the signal's symbol (may differ in multi-asset mode)
                            signal_symbol = signal.get('symbol', self.symbol)
                            signal_market_data = self.websocket.get_market_data(signal_symbol) or market_data
                            current_price = Decimal(str(signal_market_data.get('price', signal['entry_price'])))
                            
                            # Check if strategy has revalidation method (use active_strategy in multi-asset)
                            if hasattr(active_strategy, 'revalidate_signal'):
                                if not active_strategy.revalidate_signal(signal, current_price):
                                    logger.warning(f"🚫 Signal invalidated before execution - market conditions changed")
                                    await asyncio.sleep(1)
                                    continue
                            
                            # Execute trade using V3 atomic TPSL (bulk_orders with grouping='normalTpsl')
                            logger.info(f"🎯 Executing {signal['signal_type']} signal with ATOMIC TPSL")
                            
                            # PAPER TRADING: Simulate trade instead of real execution
                            if self.is_paper_trading and self.paper_trading:
                                paper_result = self.paper_trading.open_position(
                                    symbol=signal['symbol'],
                                    side='long' if signal['side'].lower() == 'buy' else 'short',
                                    size=Decimal(str(signal['size'])),
                                    entry_price=Decimal(str(signal['entry_price'])),
                                    stop_loss=Decimal(str(signal['stop_loss'])),
                                    take_profit=Decimal(str(signal['take_profit'])),
                                    leverage=int(os.getenv('MAX_LEVERAGE', '5')),
                                )
                                
                                if paper_result.get('success'):
                                    self.trades_executed += 1
                                    active_strategy.record_trade_execution(signal, {'success': True, 'paper': True})
                                    logger.info(f"📝 [PAPER] Trade executed: {signal['signal_type']} {signal['symbol']}")
                                    
                                    if self.telegram_bot:
                                        try:
                                            await self.telegram_bot.send_message(
                                                f"📝 **PAPER TRADE**\n"
                                                f"{signal['signal_type']} {signal['symbol']}\n"
                                                f"Entry: ${signal['entry_price']:.4f}\n"
                                                f"Size: {signal['size']}\n"
                                                f"SL: ${signal['stop_loss']:.4f} | TP: ${signal['take_profit']:.4f}"
                                            )
                                        except Exception:
                                            pass
                                else:
                                    logger.warning(f"📝 [PAPER] Trade failed: {paper_result.get('error')}")
                                
                                await asyncio.sleep(1)
                                continue
                            
                            # REAL TRADING: Execute actual order
                            result = await self.order_manager.place_market_order_with_stops(
                                symbol=signal['symbol'],
                                side=signal['side'].lower(),
                                size=Decimal(str(signal['size'])),
                                sl_price=Decimal(str(signal['stop_loss'])),
                                tp_price=Decimal(str(signal['take_profit'])),
                                entry_price=Decimal(str(signal['entry_price']))
                            )
                            
                            if result.get('success'):
                                self.trades_executed += 1
                                self.risk_engine.record_trade()
                                self.kill_switch.record_trade(True)
                                active_strategy.record_trade_execution(signal, result)
                                
                                # Update multi-asset manager if in multi-asset mode
                                if self.multi_asset_mode and self.asset_manager:
                                    position_side = 'long' if signal['side'].lower() == 'buy' else 'short'
                                    self.asset_manager.update_position_state(
                                        signal['symbol'],
                                        has_position=True,
                                        side=position_side,
                                        size=Decimal(str(signal['size'])),
                                        entry_price=Decimal(str(signal['entry_price']))
                                    )
                                
                                # Mark position as bot-created for tracking
                                if self.position_manager:
                                    position_side = 'long' if signal['side'].lower() == 'buy' else 'short'
                                    self.position_manager.mark_position_as_bot_created(signal['symbol'], position_side)
                                
                                # Send Telegram notification for successful fill
                                if self.telegram_bot:
                                    try:
                                        fill_data = {
                                            'symbol': signal['symbol'],
                                            'side': signal['side'],
                                            'price': signal['entry_price'],
                                            'size': signal['size'],
                                            'closed_pnl': 0  # Entry, no P&L yet
                                        }
                                        await self.telegram_bot.notify_fill(fill_data)
                                    except Exception:
                                        pass
                                
                                # Log trade for AI training (also stores trade_id in _active_trade_ids)
                                await self.log_trade_for_ai(signal, result, market_data)
                                
                                logger.info(f"✅ Trade #{self.trades_executed} executed")
                            else:
                                self.kill_switch.record_trade(False)
                                logger.warning(f"⚠️  Trade failed: {result.get('error')}")
                        else:
                            logger.info(f"🚫 Signal rejected: {rejection_reason}")
                    
                    # Reset transient error counter on successful iteration
                    if hasattr(self, '_transient_error_count') and self._transient_error_count > 0:
                        logger.info(f"✅ API connection restored after {self._transient_error_count} transient errors")
                        self._transient_error_count = 0
                    
                    # Log status every 100 loops
                    if loop_count % 100 == 0:
                        logger.info(f"📊 Loop #{loop_count} - Trades: {self.trades_executed} - P&L: ${self.session_pnl:+.2f}")
                    
                    # Loop delay
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check if this is a transient/server error (502, 503, 504, timeout, etc.)
                    is_transient = any(pattern in error_str for pattern in [
                        '502', '503', '504', '520', '521', '522', '523', '524',
                        'bad gateway', 'service unavailable', 'gateway timeout',
                        'connection reset', 'connection refused', 'timeout',
                        'temporarily unavailable', 'rate limit'
                    ])
                    
                    if is_transient:
                        # Increment transient error counter
                        if not hasattr(self, '_transient_error_count'):
                            self._transient_error_count = 0
                        self._transient_error_count += 1
                        
                        # Progressive backoff: 5s, 10s, 30s, 60s max
                        backoff_seconds = min(60, 5 * (2 ** min(3, self._transient_error_count // 5)))
                        
                        logger.warning(f"⚠️ Transient error #{self._transient_error_count}: {e} - sleeping {backoff_seconds}s")
                        
                        # Only alert every 10 transient errors
                        if self.error_handler and self._transient_error_count % 10 == 0:
                            await self.error_handler.handle_transient_error(e, "Trading Loop - API Outage")
                        
                        await asyncio.sleep(backoff_seconds)
                    else:
                        # Reset transient counter on non-transient error
                        self._transient_error_count = 0
                        
                        logger.error(f"❌ Loop error: {e}", exc_info=True)
                        
                        # Notify via error handler for real errors
                        if self.error_handler:
                            await self.error_handler.handle_critical_error(e, "Trading Loop Iteration")
                        
                        await asyncio.sleep(5)
        
        finally:
            # Stop Position Manager monitoring
            if position_manager_task:
                position_manager_task.cancel()
                try:
                    await position_manager_task
                except asyncio.CancelledError:
                    pass
            await self.shutdown()
    
    async def _run_position_manager_loop(self):
        """
        Run Position Manager in parallel with trading loop.
        Handles:
        - Detection and management of manual positions
        - Auto TP/SL setting for unmanaged positions
        - Early exit on failed setups
        """
        logger.info("🔄 Position Manager loop started")
        
        try:
            while not shutdown_event.is_set() and self.is_running:
                try:
                    # Skip if paused
                    if self.is_paused:
                        await asyncio.sleep(5)
                        continue
                    
                    # Get current candles for indicator calculation
                    candles = self._candles_cache if self._candles_cache else []
                    
                    # Scan for new/manual positions
                    new_positions = await self.position_manager.scan_positions()
                    
                    # Notify about new manual positions
                    for pos in new_positions:
                        if pos.is_manual and self.telegram_bot:
                            try:
                                await self.telegram_bot.notify_manual_position_detected(
                                    symbol=pos.symbol,
                                    side=pos.side,
                                    size=pos.size,
                                    entry_price=pos.entry_price
                                )
                            except Exception as e:
                                logger.debug(f"Failed to send manual position notification: {e}")
                    
                    # Manage each tracked position
                    for symbol, position in list(self.position_manager.positions.items()):
                        exit_reason = await self.position_manager.manage_position(position, candles)
                        
                        if exit_reason:
                            # ACTUALLY CLOSE THE POSITION!
                            logger.warning(f"🚨 Early exit triggered for {symbol}: {exit_reason.value}")
                            try:
                                # Cancel existing orders first
                                self.order_manager.cancel_all(symbol)
                                
                                # Close position with market order
                                close_result = self.order_manager.market_close(symbol)
                                if close_result.get('status') == 'ok':
                                    logger.info(f"✅ {symbol} position closed via early exit")
                                    # Remove from tracking
                                    self.position_manager.remove_position(symbol)
                                else:
                                    logger.error(f"❌ Failed to close {symbol}: {close_result}")
                            except Exception as e:
                                logger.error(f"❌ Error closing {symbol} on early exit: {e}")
                            
                            # Send Telegram notification
                            if self.telegram_bot:
                                try:
                                    await self.telegram_bot.notify_early_exit(
                                        symbol=position.symbol,
                                        side=position.side,
                                        reason=exit_reason.value,
                                        pnl=position.unrealized_pnl,
                                        health=position.health_checks_failed
                                    )
                                except Exception as e:
                                    logger.debug(f"Failed to send early exit notification: {e}")
                    
                    # Use interval from config
                    await asyncio.sleep(self.position_manager.config.get('check_interval_seconds', 5))
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Position Manager error: {e}", exc_info=True)
                    await asyncio.sleep(10)  # Back off on error
                    
        except asyncio.CancelledError:
            logger.info("🔄 Position Manager loop cancelled")
        
        logger.info("🔄 Position Manager loop stopped")
    
    async def log_trade_for_ai(self, signal: Dict, result: Dict, market_data: Dict):
        """Log trade data for AI training"""
        try:
            # Defensive: ensure signal and result are dicts
            if not isinstance(signal, dict):
                logger.warning(f"log_trade_for_ai: signal is not a dict: {type(signal)}")
                return
            if not isinstance(result, dict):
                logger.warning(f"log_trade_for_ai: result is not a dict: {type(result)}")
                return
            if not isinstance(market_data, dict):
                market_data = {}
            
            # Convert signal Decimals to float for JSON serialization
            safe_signal = {}
            for k, v in signal.items():
                if isinstance(v, Decimal):
                    safe_signal[k] = float(v)
                else:
                    safe_signal[k] = v
            
            # Convert result Decimals to float
            safe_result = {}
            for k, v in result.items():
                if isinstance(v, Decimal):
                    safe_result[k] = float(v)
                else:
                    safe_result[k] = v
            
            trade_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal': safe_signal,
                'result': safe_result,
                'market_data': market_data,
                'account_state': {
                    'equity': float(self.account_value),
                    'session_pnl': float(self.session_pnl)
                }
            }
            
            # Log to database if available
            if self.db:
                try:
                    # Map direction to database signal_type (BUY/SELL)
                    # long -> BUY (we're buying to open long)
                    # short -> SELL (we're selling to open short)
                    direction = signal.get('direction', signal.get('side', 'long'))
                    db_signal_type = 'BUY' if direction.lower() in ['long', 'buy'] else 'SELL'
                    
                    # Insert signal with indicators
                    indicators = {
                        'rsi': market_data.get('rsi'),
                        'macd': market_data.get('macd'),
                        'macd_signal': market_data.get('macd_signal'),
                        'macd_histogram': market_data.get('macd_histogram'),
                        'ema_9': market_data.get('ema_9'),
                        'ema_21': market_data.get('ema_21'),
                        'ema_50': market_data.get('ema_50'),
                        'adx': market_data.get('adx'),
                        'atr': market_data.get('atr'),
                        'volume': market_data.get('volume')
                    }
                    
                    signal_id = await self.db.insert_signal(
                        symbol=signal.get('symbol', self.symbol),
                        signal_type=db_signal_type,  # Use mapped BUY/SELL
                        price=float(signal.get('entry_price', signal.get('price', 0))),
                        confidence_score=float(signal.get('signal_score', signal.get('confidence', 0))),
                        indicators=indicators,
                        volatility=market_data.get('volatility'),
                        liquidity_score=market_data.get('liquidity_score')
                    )
                    
                    # Insert trade (check 'success' key from order execution)
                    if result.get('success'):
                        # Extract signal score (check multiple possible keys)
                        raw_score = signal.get('signal_score', signal.get('score', signal.get('confidence', 0)))
                        confidence_score = float(raw_score) if raw_score else 0.0
                        logger.debug(f"📊 Signal score for DB: {confidence_score} (raw={raw_score}, keys={list(signal.keys())[:10]})")
                        
                        trade_id = await self.db.insert_trade(
                            symbol=signal.get('symbol', self.symbol),
                            signal_type=db_signal_type,  # Use mapped BUY/SELL
                            entry_price=float(result.get('entry_price', signal.get('entry_price', 0))),
                            quantity=float(result.get('quantity', signal.get('size', 0))),
                            confidence_score=confidence_score,
                            strategy_name=signal.get('strategy'),
                            account_equity=float(self.account_value),
                            session_pnl=float(self.session_pnl),
                            order_id=result.get('order_id')
                        )
                        
                        # Link signal to trade
                        await self.db.mark_signal_executed(signal_id, trade_id)
                        
                        # Store trade_id for closing later when position closes
                        symbol = signal.get('symbol', self.symbol)
                        self._active_trade_ids[symbol] = {
                            'trade_id': trade_id,
                            'entry_price': float(result.get('entry_price', signal.get('entry_price', 0))),
                            'quantity': float(result.get('quantity', signal.get('size', 0))),
                            'side': direction,  # Store actual direction (long/short)
                            'signal_type': db_signal_type,  # Store BUY/SELL for DB
                            'entry_time': datetime.now(timezone.utc)
                        }
                        
                        logger.info(f"📊 Logged to database: signal_id={signal_id}, trade_id={trade_id}, type={db_signal_type}")
                    else:
                        # Signal rejected
                        await self.db.mark_signal_rejected(
                            signal_id,
                            result.get('rejection_reason', 'Unknown')
                        )
                        logger.info(f"📊 Signal #{signal_id} rejected: {result.get('rejection_reason')}")
                        
                except Exception as db_error:
                    logger.error(f"Database logging failed: {db_error}")
                    # Fall through to JSONL backup
            
            # Backup to JSONL (for now, can remove later)
            log_file = self.trade_log_path / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down...")
        self.is_running = False
        
        try:
            # Stop Telegram bot
            if self.telegram_bot:
                await self.telegram_bot.stop()
            
            # Stop websocket (sync method)
            if self.websocket:
                self.websocket.stop()
            
            # Close database connection
            if self.db:
                await self.db.disconnect()
            
            # Log final statistics
            runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
            logger.info("📊 Session Summary:")
            logger.info(f"   Runtime: {runtime/3600:.2f} hours")
            logger.info(f"   Trades: {self.trades_executed}")
            logger.info(f"   Session P&L: ${self.session_pnl:+.2f}")
            logger.info(f"   Final Equity: ${self.account_value:.2f}")
            
            if self.strategy:
                stats = self.strategy.get_statistics()
                logger.info(f"   Strategy Stats: {stats}")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def pause(self):
        """Pause trading (for emergency stop button)"""
        if not self.is_paused:
            self.is_paused = True
            logger.warning("⏸️ TRADING PAUSED - No new positions will be opened")
    
    def resume(self):
        """Resume trading (for start button)"""
        if self.is_paused:
            self.is_paused = False
            logger.info("▶️ TRADING RESUMED - Bot is active")
    
    async def get_account_status(self) -> Dict[str, Any]:
        """Get current account status for Telegram"""
        return {
            'account_value': float(self.account_value),
            'margin_used': float(self.margin_used),
            'withdrawable': float(self.account_value - self.margin_used),
            'session_pnl': float(self.session_pnl),
            'trades_executed': self.trades_executed,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'uptime': (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        }


async def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("🤖 HYPERAI TRADER - Starting")
    logger.info("="*60)
    
    bot = HyperAIBot()
    
    if await bot.initialize():
        await bot.run_trading_loop()
    else:
        logger.error("❌ Failed to initialize bot")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
