"""
HyperLiquid WebSocket - Ultra-Lean SDK Passthrough
Uses SDK info.subscribe() for all real-time data.
"""
import os
import asyncio
import threading
from typing import Optional, Dict, Any, Callable, List, Set
from hyperliquid.info import Info
from hyperliquid.utils import constants
from app.utils.trading_logger import TradingLogger

logger = TradingLogger("hl_websocket")


class HLWebSocket:
    """
    Ultra-lean WebSocket using SDK info.subscribe().
    
    Subscription types (SDK native):
    - allMids: All mid prices
    - l2Book: Order book depth
    - trades: Trade feed
    - candle: Candlestick data
    - userEvents: All user events (fills, orders, funding)
    - userFills: User fill events only
    - orderUpdates: Order status updates
    - userFundings: Funding payments
    - bbo: Best bid/offer
    """
    
    def __init__(self, address: str, testnet: bool = False):
        self.address = address
        
        # Get RPC URL from env or use SDK defaults
        if testnet:
            base_url = os.getenv('TESTNET_RPC_URL', constants.TESTNET_API_URL)
        else:
            base_url = os.getenv('MAINNET_RPC_URL', constants.MAINNET_API_URL)
        
        # SDK Info with WebSocket enabled
        self.info = Info(base_url, skip_ws=False)
        
        # Callbacks by subscription type
        self._callbacks: Dict[str, List[Callable]] = {}
        self._active_subs: Set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Market data cache (populated by subscriptions)
        self._market_data: Dict[str, Dict[str, Any]] = {}
        self._all_mids: Dict[str, float] = {}
    
    # ==================== CALLBACK MANAGEMENT ====================
    def add_callback(self, sub_type: str, callback: Callable):
        """Register callback for subscription type."""
        if sub_type not in self._callbacks:
            self._callbacks[sub_type] = []
        self._callbacks[sub_type].append(callback)
    
    def remove_callback(self, sub_type: str, callback: Callable):
        """Remove callback for subscription type."""
        if sub_type in self._callbacks and callback in self._callbacks[sub_type]:
            self._callbacks[sub_type].remove(callback)
    
    def _dispatch(self, sub_type: str, data: Any):
        """Dispatch data to registered callbacks."""
        for cb in self._callbacks.get(sub_type, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error for {sub_type}: {e}")
    
    # ==================== SUBSCRIPTIONS ====================
    def subscribe_all_mids(self, callback: Optional[Callable] = None):
        """Subscribe to all mid prices."""
        def _handle_mids(msg):
            # Update internal cache
            if isinstance(msg, dict) and "mids" in msg:
                self._all_mids = {k: float(v) for k, v in msg.get("mids", {}).items()}
                # Update market data for each symbol
                for symbol, price in self._all_mids.items():
                    if symbol not in self._market_data:
                        self._market_data[symbol] = {}
                    self._market_data[symbol]["price"] = price
            self._dispatch("allMids", msg)
        
        if callback:
            self.add_callback("allMids", callback)
        self.info.subscribe({"type": "allMids"}, _handle_mids)
        self._active_subs.add("allMids")
        logger.info("Subscribed to allMids")
    
    def subscribe_l2_book(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to order book for symbol."""
        sub_key = f"l2Book:{symbol}"
        if callback:
            self.add_callback(sub_key, callback)
        self.info.subscribe(
            {"type": "l2Book", "coin": symbol},
            lambda msg: self._dispatch(sub_key, msg)
        )
        self._active_subs.add(sub_key)
        logger.info(f"Subscribed to l2Book:{symbol}")
    
    def subscribe_trades(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to trade feed for symbol."""
        sub_key = f"trades:{symbol}"
        if callback:
            self.add_callback(sub_key, callback)
        self.info.subscribe(
            {"type": "trades", "coin": symbol},
            lambda msg: self._dispatch(sub_key, msg)
        )
        self._active_subs.add(sub_key)
        logger.info(f"Subscribed to trades:{symbol}")
    
    def subscribe_candles(self, symbol: str, interval: str = "1m", 
                          callback: Optional[Callable] = None):
        """
        Subscribe to candlestick data.
        Intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
        """
        sub_key = f"candle:{symbol}:{interval}"
        
        def _handle_candle(msg: Dict):
            """Handle candle message and dispatch with (symbol, candle) signature."""
            try:
                # Extract candle data from message
                data = msg.get("data", msg)
                if isinstance(data, list) and len(data) > 0:
                    candle = data[-1]  # Latest candle
                elif isinstance(data, dict):
                    candle = data
                else:
                    candle = msg
                
                # Dispatch with symbol for callbacks expecting (symbol, candle)
                for cb in self._callbacks.get(sub_key, []):
                    try:
                        cb(symbol, candle)
                    except TypeError:
                        # Callback doesn't expect symbol, just pass candle
                        cb(candle)
            except Exception as e:
                logger.error(f"Callback error for {sub_key}: {e}")
        
        if callback:
            self.add_callback(sub_key, callback)
        self.info.subscribe(
            {"type": "candle", "coin": symbol, "interval": interval},
            _handle_candle
        )
        self._active_subs.add(sub_key)
        logger.info(f"Subscribed to candle:{symbol}:{interval}")
    
    def subscribe_user_events(self, callback: Optional[Callable] = None):
        """Subscribe to all user events (fills, orders, funding)."""
        if callback:
            self.add_callback("userEvents", callback)
        self.info.subscribe(
            {"type": "userEvents", "user": self.address},
            lambda msg: self._dispatch("userEvents", msg)
        )
        self._active_subs.add("userEvents")
        logger.info("Subscribed to userEvents")
    
    def subscribe_user_fills(self, callback: Optional[Callable] = None):
        """Subscribe to user fill events only."""
        if callback:
            self.add_callback("userFills", callback)
        self.info.subscribe(
            {"type": "userFills", "user": self.address},
            lambda msg: self._dispatch("userFills", msg)
        )
        self._active_subs.add("userFills")
        logger.info("Subscribed to userFills")
    
    def subscribe_order_updates(self, callback: Optional[Callable] = None):
        """Subscribe to order status updates."""
        if callback:
            self.add_callback("orderUpdates", callback)
        self.info.subscribe(
            {"type": "orderUpdates", "user": self.address},
            lambda msg: self._dispatch("orderUpdates", msg)
        )
        self._active_subs.add("orderUpdates")
        logger.info("Subscribed to orderUpdates")
    
    def subscribe_user_fundings(self, callback: Optional[Callable] = None):
        """Subscribe to funding payments."""
        if callback:
            self.add_callback("userFundings", callback)
        self.info.subscribe(
            {"type": "userFundings", "user": self.address},
            lambda msg: self._dispatch("userFundings", msg)
        )
        self._active_subs.add("userFundings")
        logger.info("Subscribed to userFundings")
    
    def subscribe_bbo(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to best bid/offer for symbol."""
        sub_key = f"bbo:{symbol}"
        if callback:
            self.add_callback(sub_key, callback)
        self.info.subscribe(
            {"type": "bbo", "coin": symbol},
            lambda msg: self._dispatch(sub_key, msg)
        )
        self._active_subs.add(sub_key)
        logger.info(f"Subscribed to bbo:{symbol}")
    
    # ==================== LIFECYCLE ====================
    def start(self):
        """Start WebSocket processing in background thread."""
        if self._running:
            return
        self._running = True
        logger.info("WebSocket started")
    
    def stop(self):
        """Stop WebSocket and cleanup."""
        self._running = False
        self._active_subs.clear()
        self._callbacks.clear()
        logger.info("WebSocket stopped")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is active."""
        return self._running and len(self._active_subs) > 0
    
    @property
    def active_subscriptions(self) -> Set[str]:
        """Get list of active subscriptions."""
        return self._active_subs.copy()
    
    # ==================== BOT.PY COMPATIBILITY ALIASES ====================
    def add_candle_callback(self, callback: Callable):
        """Alias for bot.py compatibility - adds candle callback."""
        self.add_callback("candle", callback)
    
    def add_order_update_callback(self, callback: Callable):
        """Alias for bot.py compatibility - adds order update callback."""
        self.add_callback("orderUpdates", callback)
    
    def add_fill_callback(self, callback: Callable):
        """Alias for bot.py compatibility - adds fill callback."""
        self.add_callback("userFills", callback)
    
    def get_cached_state(self) -> Optional[Dict]:
        """Get cached account state (for client integration)."""
        # This would be populated by userEvents subscription
        return None  # Not implemented - falls back to API
    
    # ==================== MARKET DATA ACCESS ====================
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get cached market data for symbol.
        
        Returns dict with:
        - price: Current mid price
        - bid: Best bid (if l2Book subscribed)
        - ask: Best ask (if l2Book subscribed)
        - volume: 24h volume (if available)
        """
        # First try symbol-specific cache
        data = self._market_data.get(symbol, {}).copy()
        
        # Fallback to allMids for price
        if "price" not in data and symbol in self._all_mids:
            data["price"] = self._all_mids[symbol]
        
        # If still no price, try fetching from info API
        if "price" not in data:
            try:
                mids = self.info.all_mids()
                if symbol in mids:
                    data["price"] = float(mids[symbol])
                    self._all_mids[symbol] = data["price"]
            except Exception as e:
                logger.error(f"Failed to fetch mids: {e}")
        
        return data if data else {"price": None}
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all cached mid prices."""
        return self._all_mids.copy()


# Factory function
def create_websocket(address: str, testnet: bool = False) -> HLWebSocket:
    """Create HLWebSocket instance."""
    return HLWebSocket(address, testnet)
