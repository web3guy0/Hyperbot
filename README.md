# 🤖 HyperBot - Automated Trading Bot for HyperLiquid DEX

[![Production Ready](https://img.shields.io/badge/status-production%20ready-success)](https://github.com/web3guy0/hyperbot)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Enterprise-grade automated trading bot** combining rule-based strategies with adaptive risk management for cryptocurrency futures trading on HyperLiquid DEX.

## 🆕 Version 5.0 - Institutional Trading Features

**Major upgrade with institutional-grade enhancements:**
- **Human-Like Trading Intelligence** - Anti-chase, mean reversion, stop hunt detection
- **Multi-Timeframe Confirmation** - Mandatory 15m/1h/4h alignment before entries
- **Smart Money Concepts** - FVG, Order Blocks, Liquidity Sweeps, Break of Structure
- **Adaptive Risk Management** - ATR-based dynamic TP/SL with regime adjustment
- **Kelly Criterion Sizing** - Optimal position sizing based on win rate
- **Multi-Asset Trading** - Trade SOL, ETH, BTC simultaneously
- **Paper Trading Mode** - Validate strategies without real money
- **Small Account Mode** - Optimized for $20-$100 accounts

---

## 📁 Project Structure

```
Hyperbot/
├── app/
│   ├── bot.py                    # Main bot orchestrator
│   ├── config.py                 # Configuration management
│   ├── backtesting/
│   │   └── backtester.py         # Historical backtesting engine
│   ├── database/
│   │   ├── db_manager.py         # PostgreSQL database manager
│   │   ├── analytics.py          # Performance analytics dashboard
│   │   └── schema.sql            # Database schema
│   ├── execution/
│   │   └── paper_trading.py      # Paper trading simulator
│   ├── hl/
│   │   ├── hl_client.py          # HyperLiquid API client
│   │   ├── hl_order_manager.py   # Order execution & management
│   │   └── hl_websocket.py       # Real-time WebSocket feeds
│   ├── portfolio/
│   │   ├── multi_asset_manager.py # Multi-asset orchestration
│   │   └── position_manager.py    # Position tracking
│   ├── risk/
│   │   ├── drawdown_monitor.py   # Drawdown tracking
│   │   ├── kelly_criterion.py    # Optimal position sizing
│   │   ├── kill_switch.py        # Emergency stop logic
│   │   ├── risk_engine.py        # Core risk management
│   │   └── small_account_mode.py # Small account optimizations
│   ├── strategies/
│   │   ├── strategy_manager.py   # Strategy orchestration
│   │   ├── adaptive/             # Adaptive strategy components
│   │   │   ├── adaptive_risk.py      # Dynamic TP/SL calculator
│   │   │   ├── cmf.py                # Chaikin Money Flow
│   │   │   ├── divergence.py         # RSI/MACD divergence
│   │   │   ├── donchian.py           # Donchian channels
│   │   │   ├── funding_rate.py       # Funding rate filter
│   │   │   ├── market_regime.py      # Regime detection
│   │   │   ├── multi_asset_correlation.py # BTC correlation
│   │   │   ├── multi_timeframe.py    # MTF analysis
│   │   │   ├── obv.py                # On-Balance Volume
│   │   │   ├── order_flow.py         # Order flow analysis
│   │   │   ├── pro_filters.py        # Professional filters
│   │   │   ├── session_manager.py    # Trading session detection
│   │   │   ├── smart_money.py        # SMC concepts
│   │   │   ├── stoch_rsi.py          # Stochastic RSI
│   │   │   ├── supertrend.py         # Supertrend indicator
│   │   │   └── vwap.py               # VWAP calculator
│   │   └── rule_based/
│   │       └── swing_strategy.py # Main swing trading strategy
│   ├── tg_bot/
│   │   ├── bot.py                # Telegram bot
│   │   ├── formatters.py         # Message formatting
│   │   └── keyboards.py          # Interactive buttons
│   └── utils/
│       ├── error_handler.py      # Error handling
│       ├── health_check.py       # 🆕 HTTP health check server
│       ├── indicator_calculator.py # Shared indicator calculator
│       ├── position_calculator.py # Position calculations
│       ├── symbol_manager.py     # Symbol management
│       └── trading_logger.py     # Logging utilities
├── tests/                        # 🆕 Pytest unit tests
│   ├── conftest.py               # Test fixtures
│   ├── test_indicators.py        # RSI, EMA, ATR, ADX tests
│   ├── test_pnl_calculations.py  # PnL, TP/SL, position sizing
│   ├── test_risk_management.py   # Risk limits, drawdown, kill switch
│   └── test_signals.py           # Anti-chase, RSI blocks, scoring
├── ml/
│   ├── auto_trainer.py           # ML auto-retraining (future)
│   ├── training/
│   │   ├── dataset_builder.py    # Training data preparation
│   │   ├── feature_engineering.py # Feature engineering
│   │   └── model_trainer.py      # Model training
│   ├── evaluation/               # Model evaluation (placeholder)
│   ├── inference/                # Model inference (placeholder)
│   └── models/saved/             # Saved models
├── scripts/
│   └── backfill_trades.py        # Database backfill utility
├── data/
│   ├── bot_positions.json        # Position state persistence
│   └── trades/                   # Trade logs (JSONL)
├── logs/                         # Application logs
├── ecosystem.config.js           # PM2 process manager config
├── hyperbot.service              # Systemd service file
├── pyrightconfig.json            # Type checking config
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## ⚡ Quick Start

### **1. Clone & Install**
```bash
git clone https://github.com/web3guy0/hyperbot.git
cd hyperbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **2. Configure**
```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required settings (5 minimum):**
```env
# HyperLiquid API (REQUIRED)
API_SECRET=0x...              # Your wallet private key
ACCOUNT_ADDRESS=0x...         # Your wallet address

# Database (REQUIRED)
DATABASE_URL=postgresql://... # PostgreSQL connection string

# Telegram (REQUIRED for notifications)
TELEGRAM_BOT_TOKEN=...        # From @BotFather
TELEGRAM_CHAT_ID=...          # Your chat ID
```

### **3. Start Trading**
```bash
# Paper trading first (recommended)
PAPER_TRADING=true python -m app.bot

# Testnet
TESTNET=true python -m app.bot

# Production with PM2
pm2 start ecosystem.config.js
pm2 logs hyperbot
```

---

## 🧠 Human-Like Trading Intelligence

### **The Problem: Why Bots Lose Money**
Most trading bots make the same mistakes retail traders make:
1. **Chasing momentum** - Buying after 3+ green candles (too late!)
2. **Getting stopped out** - SL placed at obvious levels (stop hunts)
3. **Fighting the trend** - Mean reverting in trending markets

### **Our Solution: HumanTradingLogic Module**

```
HUMAN-LIKE TRADING INTELLIGENCE
═══════════════════════════════════════════════════════════════

1. Anti-Chase Logic
├─ Detects "chasing" (3+ consecutive same-color candles)
├─ PENALIZES signals that follow momentum
├─ REWARDS counter-momentum entries
└─ Result: Enter BEFORE the crowd, not after

2. Mean Reversion Detection
├─ RSI extremes (<25 or >75)
├─ Extended moves (>2% from VWAP)
├─ Bollinger Band touches
└─ Result: Catch reversals at key levels

3. Liquidity Sweep Detection
├─ Identifies stop hunt patterns
├─ Detects "sweep + reversal" setups
├─ Confirms with volume surge
└─ Result: Trade WITH smart money, not against

4. Smart Stop Placement
├─ Places SL BEYOND obvious liquidity pools
├─ Uses ATR + SMC for dynamic levels
├─ Avoids round numbers where stops cluster
└─ Result: Fewer stops hit by wicks

═══════════════════════════════════════════════════════════════
```

**Configuration:**
```env
USE_HUMAN_LOGIC=true          # Enable human-like logic
HUMAN_LOGIC_WEIGHT=2.0        # Weight for human signals (1.0-3.0)
```

---

## 🎯 How SL/TP is Calculated (Pro Trader Style)

### **The Problem with Fixed Percentages**
Amateur bots use fixed SL/TP like "stop at -1%, profit at +2%". This FAILS because:
- Volatility changes (1% in calm market ≠ 1% in volatile market)
- No consideration of market structure
- Stops get hit by normal price noise

### **Our Pro Approach: ATR + SMC + Regime**

```
SL/TP CALCULATION LAYERS
═══════════════════════════════════════════════════════════════

Layer 1: ATR-Based Dynamic Levels
├─ Stop Loss = Entry ± (ATR × 1.2-1.5)
├─ Take Profit = Entry ± (ATR × 4.0-4.5)
└─ Result: Levels adapt to current volatility

   Example (SOL at $200, ATR = $3):
   • SL Distance = $3 × 1.2 = $3.60 (1.8%)
   • TP Distance = $3 × 4.5 = $13.50 (6.75%)
   • R:R Ratio = 3.75:1

Layer 2: Market Regime Adjustment
├─ TRENDING: TP×2.5, SL×0.8 (ride the trend)
├─ RANGING: TP×1.2, SL×0.8 (quick scalps)
├─ VOLATILE: TP×3.0, SL×1.5 (wider everything)
├─ BREAKOUT: TP×4.0, SL×0.5 (big move potential)
└─ Result: Adapts to market conditions

Layer 3: Smart Money Concepts (SMC)
├─ Fair Value Gaps (FVG) - Unmitigated price imbalances
├─ Order Blocks - Where institutions accumulated
├─ Liquidity Levels - Where stop losses cluster
└─ Result: TP/SL placed at institutional levels

Layer 4: Liquidity Targeting
├─ Identify where stops cluster (swing highs/lows)
├─ Set TP before liquidity pools
├─ Set SL beyond liquidity sweeps
└─ Result: Exit before reversals, avoid stop hunts

═══════════════════════════════════════════════════════════════
```

---

## 🔬 Multi-Timeframe Confirmation (Mandatory)

Every signal MUST align across timeframes:

```
ENTRY CONFIRMATION FLOW
═══════════════════════════════════════════════════════════════

4H Timeframe (Structure)
├─ Trend direction: UP / DOWN / RANGING
├─ Key S/R levels identified
└─ Bias: Only trade WITH 4H trend

           ↓

1H Timeframe (Momentum)  
├─ Confirm 4H direction
├─ RSI not overbought/oversold
├─ EMA alignment (21 > 50 for longs)
└─ Check: If 4H=UP, 1H must also be UP

           ↓

15M Timeframe (Entry Zone)
├─ Refine entry timing
├─ Look for pullback entries
├─ Confirm momentum with MACD
└─ Check: Must align with 1H and 4H

           ↓

1M/5M Timeframe (Execution)
├─ Precise entry trigger
├─ FVG or Order Block touch
├─ Tight SL placement
└─ EXECUTE only if all TFs align!

═══════════════════════════════════════════════════════════════
```

---

## 📊 Signal Scoring System

The bot uses a comprehensive 10-point scoring system:

| Component | Max Points | What It Measures |
|-----------|------------|------------------|
| Technical Indicators | 4 | RSI, MACD, EMA, Bollinger |
| SMC Alignment | 2 | FVG, Order Blocks, Liquidity |
| HTF Alignment | 2 | 15m/1h/4h trend agreement |
| Order Flow | 2 | Volume delta, aggressive buying/selling |

**Entry Threshold: 7/10 minimum** (configurable via `MIN_SIGNAL_SCORE`)

**Score Adjustments:**
- Human Logic can add +2 or -2 based on market context
- Ranging markets get -2 threshold reduction
- High volatility adds +1 threshold increase

---

## 🛡️ Risk Management Architecture

### **Multi-Layer Protection**

```
RISK MANAGEMENT STACK
═══════════════════════════════════════════════════════════════

1. Kelly Criterion (Position Sizing)
   ├─ Calculates optimal bet size: f* = (p×b - q) / b
   ├─ Uses Half-Kelly for safety (0.5× recommended)
   ├─ Tracks last 20 trades for win rate
   └─ Adapts position size to performance

2. Adaptive Risk Manager (TP/SL)
   ├─ ATR-based dynamic levels
   ├─ Regime-adjusted multipliers
   ├─ 20-trade rolling performance tracking
   ├─ Reduces risk after consecutive losses
   └─ Session-aware adjustments

3. Kill Switch (Emergency Stop)
   ├─ Daily loss limit: -5% (configurable)
   ├─ Max drawdown: -10% from peak
   ├─ Single position loss: -8%
   └─ Auto-pause trading when triggered

4. Drawdown Monitor
   ├─ Tracks peak equity
   ├─ Calculates current drawdown
   ├─ Alerts at warning thresholds
   └─ Forces stop at max drawdown

═══════════════════════════════════════════════════════════════
```

---

## 💰 Small Account Mode ($20-$100)

Automatically activated for accounts under $100:

```env
SMALL_ACCOUNT_MODE=auto       # auto, true, or false
SMALL_ACCOUNT_THRESHOLD=100   # Threshold in USD
```

**Optimizations:**
- Leverage: 10x (vs 5x default)
- Position Size: 80% of balance
- Tighter SL to preserve capital
- Best assets: SOL, ETH (lower fees)

---

## 📝 Paper Trading Mode

Test strategies without risking real money:

```bash
PAPER_TRADING=true PAPER_TRADING_BALANCE=1000 python -m app.bot
```

Features:
- Full strategy execution (simulated)
- Track virtual P&L
- Performance metrics
- No real orders sent

---

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Bot status, account balance, uptime |
| `/positions` | Active positions with live P&L |
| `/trades` | Recent completed trades |
| `/pnl` | Daily and weekly P&L |
| `/stats` | Strategy performance stats |
| `/analytics` | Full performance dashboard |
| `/kelly` | Kelly criterion sizing info |
| `/assets` | Multi-asset status |
| `/backtest` | Run strategy backtest |
| `/logs` | Recent bot logs |
| `/help` | All available commands |

**Control Buttons:**
- 🚀 **START** - Resume trading
- 🛑 **STOP** - Pause trading
- ❌ **CLOSE ALL** - Emergency close

---

## 🔧 Configuration Reference

### **Essential Settings (5 Required)**
```env
# API (REQUIRED)
API_SECRET=0x...
ACCOUNT_ADDRESS=0x...

# Database (REQUIRED)
DATABASE_URL=postgresql://...

# Telegram (REQUIRED)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### **Trading Settings**
```env
SYMBOL=BTC                    # Primary symbol
MULTI_ASSET_MODE=true         # Enable multi-asset
MULTI_ASSETS=BTC,ETH,SOL      # Assets to trade
MAX_POSITIONS=3               # Max concurrent positions
MAX_LEVERAGE=5                # Maximum leverage
POSITION_SIZE_PCT=50          # Base position size %
```

### **Signal Quality**
```env
MIN_SIGNAL_SCORE=7            # Minimum score (1-10)
SIGNAL_CONFIRMATION_SCANS=3   # Confirmations needed
SWING_COOLDOWN=600            # Seconds between signals
```

### **Risk Management**
```env
RISK_PER_TRADE_PCT=2.0        # Risk per trade
MAX_DAILY_LOSS_PCT=5          # Daily loss limit
MAX_DRAWDOWN_PCT=10           # Max drawdown
ATR_SL_MULTIPLIER=1.2         # SL = ATR × multiplier
ATR_TP_MULTIPLIER=4.5         # TP = ATR × multiplier
```

### **Human Logic**
```env
USE_HUMAN_LOGIC=true          # Enable human-like logic
HUMAN_LOGIC_WEIGHT=2.0        # Signal weight (1.0-3.0)
```

### **Kelly Criterion**
```env
KELLY_ENABLED=true
KELLY_FRACTION=0.5            # Half-Kelly for safety
```

---

## 🧪 Testing

The bot includes a comprehensive test suite with 67+ tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_indicators.py -v
python -m pytest tests/test_pnl_calculations.py -v
python -m pytest tests/test_signals.py -v
python -m pytest tests/test_risk_management.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

---

## 🏥 Health Check API

Built-in HTTP health check server for monitoring:

```bash
# Configure port in .env
HEALTH_CHECK_PORT=8080
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe (200 if running) |
| `GET /ready` | Readiness probe (200 if trading ready) |
| `GET /status` | Detailed status with metrics |
| `GET /metrics` | Prometheus-compatible metrics |

```bash
# Example health check
curl http://localhost:8080/health
{"status": "healthy", "heartbeat_age_seconds": 5.2}

# Example status check
curl http://localhost:8080/status
{"uptime_human": "2d 5h 30m", "trades_executed": 45, ...}
```

---

## 🗄️ Database Schema

The bot uses PostgreSQL with these tables:

| Table | Purpose |
|-------|---------|
| `trades` | Completed trade history |
| `signals` | Generated signals |
| `ml_predictions` | ML model predictions (future) |
| `account_snapshots` | Account balance history |
| `performance_metrics` | Performance tracking |

---

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| Win Rate | 65-70% |
| Risk-Reward | 3:1+ |
| Daily Target | +1-3% |
| Max Daily Loss | -5% (kill switch) |
| Max Drawdown | -10% |
| Trades/Day | 3-10 (quality focused) |

---

## 🔐 Security

- ✅ API keys masked in logs
- ✅ Secrets hidden in Telegram
- ✅ No sensitive data in git
- ✅ Dedicated API wallet recommended

---

## ⚠️ Disclaimer

This bot is a **trading tool**, not financial advice:
- Cryptocurrency trading is highly risky
- Past performance ≠ future results
- Only trade with capital you can afford to lose
- Monitor the bot regularly
- Start with testnet/paper trading first

---

## 📈 Getting Started Guide

1. **Test on Paper First** - `PAPER_TRADING=true`
2. **Try Testnet** - `TESTNET=true`
3. **Start Small** - $50-100 on mainnet
4. **Monitor Daily** - Check Telegram
5. **Scale Gradually** - Increase position size slowly

---

**Version**: 5.2 (Professional Grade + Test Suite)  
**Last Updated**: January 1, 2026  
**License**: MIT

**⚡ Ready to trade like an institution? Let's go! 🚀**
