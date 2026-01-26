"""
Trailing Stop Manager - Clean Step-Based Trailing SL Logic

This module handles all trailing stop logic with clear, step-based levels:
- TP: Fixed at target (e.g., +20% ROE)
- SL: Initial at -8% ROE, then trails as profit grows

Trailing Steps (ROE-based with 10x leverage):
- PnL > +4%:  Move SL to breakeven (0%)
- PnL > +8%:  Move SL to +4%
- PnL > +12%: Move SL to +8%
- PnL > +16%: Move SL to +12%

Key Features:
1. Step-based (not continuous) - only moves at defined thresholds
2. Tracks current SL level to prevent duplicate orders
3. Cancels old SL before placing new one
4. Signal revalidation for early exit if signal becomes invalid
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SLLevel(Enum):
    """Stop Loss levels (ROE percentages)"""
    INITIAL = -8      # Initial SL at -8% ROE
    BREAKEVEN = 0     # At entry price
    LEVEL_1 = 4       # Lock +4% ROE
    LEVEL_2 = 8       # Lock +8% ROE
    LEVEL_3 = 12      # Lock +12% ROE


@dataclass
class TrailingConfig:
    """Configuration for trailing stop behavior"""
    # TP/SL targets (ROE %)
    tp_roe_pct: float = 20.0      # Fixed TP at +20% ROE
    initial_sl_roe_pct: float = -8.0  # Initial SL at -8% ROE
    
    # Trailing step triggers (when PnL reaches these, move SL)
    step_triggers: Dict[float, float] = field(default_factory=lambda: {
        4.0: 0.0,    # At +4% ROE -> Move SL to breakeven
        8.0: 4.0,    # At +8% ROE -> Move SL to +4%
        12.0: 8.0,   # At +12% ROE -> Move SL to +8%
        16.0: 12.0,  # At +16% ROE -> Move SL to +12%
    })
    
    # Signal revalidation
    revalidate_interval_sec: float = 30.0  # Revalidate signal every 30 seconds
    exit_on_invalid_signal: bool = True    # Exit if signal becomes invalid
    
    # Throttle to prevent spam
    min_update_interval_sec: float = 5.0   # Minimum time between SL updates


@dataclass
class PositionState:
    """Tracks the state of a position for trailing purposes"""
    symbol: str
    entry_price: float
    size: float
    is_long: bool
    leverage: float
    
    # Current orders
    current_sl_roe: float = -8.0   # Current SL level (ROE %)
    current_tp_roe: float = 20.0   # Current TP level (ROE %)
    sl_order_placed: bool = False
    tp_order_placed: bool = False
    
    # Tracking
    highest_roe_reached: float = 0.0  # Track highest ROE for step determination
    last_sl_update: Optional[datetime] = None
    last_signal_check: Optional[datetime] = None
    
    # Signal state
    entry_signal_score: float = 0.0
    last_signal_valid: bool = True


class TrailingManager:
    """
    Manages trailing stop logic for all positions.
    
    Usage:
        manager = TrailingManager(order_manager, config)
        
        # When position opens:
        manager.register_position(symbol, entry_price, size, is_long, leverage)
        
        # On each monitoring tick:
        await manager.update(symbol, current_pnl_roe, current_price)
        
        # On position close:
        manager.unregister_position(symbol)
    """
    
    def __init__(self, order_manager, strategy=None, config: Optional[TrailingConfig] = None):
        """
        Initialize trailing manager.
        
        Args:
            order_manager: HLOrderManager instance for placing/cancelling orders
            strategy: Strategy instance for signal revalidation (optional)
            config: Trailing configuration
        """
        self.order_manager = order_manager
        self.strategy = strategy
        self.config = config or TrailingConfig()
        
        # Track positions
        self.positions: Dict[str, PositionState] = {}
        
        # Sorted trigger levels for efficient lookup
        self._sorted_triggers = sorted(self.config.step_triggers.keys())
        
        logger.info(f"📊 TrailingManager initialized")
        logger.info(f"   TP: +{self.config.tp_roe_pct}% ROE (fixed)")
        logger.info(f"   Initial SL: {self.config.initial_sl_roe_pct}% ROE")
        logger.info(f"   Trailing steps: {self.config.step_triggers}")
    
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        is_long: bool,
        leverage: float,
        signal_score: float = 0.0
    ) -> PositionState:
        """
        Register a new position for trailing management.
        
        Args:
            symbol: Trading pair
            entry_price: Position entry price
            size: Position size
            is_long: True if long, False if short
            leverage: Position leverage
            signal_score: Entry signal score for revalidation
            
        Returns:
            PositionState for the registered position
        """
        state = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            size=size,
            is_long=is_long,
            leverage=leverage,
            current_sl_roe=self.config.initial_sl_roe_pct,
            current_tp_roe=self.config.tp_roe_pct,
            entry_signal_score=signal_score
        )
        
        self.positions[symbol] = state
        logger.info(f"📝 Registered {symbol} for trailing: Entry=${entry_price:.4f}, "
                   f"{'LONG' if is_long else 'SHORT'}, {leverage}x")
        
        return state
    
    def unregister_position(self, symbol: str) -> None:
        """Remove position from trailing management."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"📝 Unregistered {symbol} from trailing")
    
    def get_sl_price_for_roe(self, state: PositionState, target_roe: float) -> float:
        """
        Calculate SL price that achieves target ROE at given leverage.
        
        For LONG: SL below entry means loss, SL above entry means profit lock
        For SHORT: SL above entry means loss, SL below entry means profit lock
        
        ROE = (price_change / entry_price) * leverage * 100
        price_change = ROE * entry_price / (leverage * 100)
        
        Args:
            state: Position state
            target_roe: Target ROE percentage for SL
            
        Returns:
            SL price
        """
        # Convert ROE to price change
        price_change_pct = target_roe / state.leverage  # ROE / leverage = price %
        price_change = state.entry_price * (price_change_pct / 100)
        
        if state.is_long:
            # Long: + price change = profit, - price change = loss
            sl_price = state.entry_price + price_change
        else:
            # Short: - price change = profit, + price change = loss
            sl_price = state.entry_price - price_change
        
        return sl_price
    
    def get_tp_price_for_roe(self, state: PositionState, target_roe: float) -> float:
        """
        Calculate TP price that achieves target ROE at given leverage.
        
        Args:
            state: Position state
            target_roe: Target ROE percentage for TP
            
        Returns:
            TP price
        """
        price_change_pct = target_roe / state.leverage
        price_change = state.entry_price * (price_change_pct / 100)
        
        if state.is_long:
            tp_price = state.entry_price + price_change
        else:
            tp_price = state.entry_price - price_change
        
        return tp_price
    
    def get_next_sl_level(self, current_roe: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Determine the next SL level based on current ROE.
        
        Args:
            current_roe: Current ROE percentage
            
        Returns:
            Tuple of (trigger_roe, new_sl_roe) or (None, None) if no update needed
        """
        # Find the highest trigger we've crossed
        highest_trigger = None
        new_sl = None
        
        for trigger in self._sorted_triggers:
            if current_roe >= trigger:
                highest_trigger = trigger
                new_sl = self.config.step_triggers[trigger]
        
        return highest_trigger, new_sl
    
    async def update(
        self,
        symbol: str,
        current_roe: float,
        current_price: float,
        candles: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Update trailing stop for a position.
        
        This is the main method called on each monitoring tick.
        
        Args:
            symbol: Trading pair
            current_roe: Current ROE percentage
            current_price: Current market price
            candles: Recent candles for signal revalidation (optional)
            
        Returns:
            Dict with update status and any actions taken
        """
        if symbol not in self.positions:
            return {'status': 'not_tracked'}
        
        state = self.positions[symbol]
        now = datetime.now(timezone.utc)
        result = {'status': 'ok', 'actions': []}
        
        # Track highest ROE reached
        if current_roe > state.highest_roe_reached:
            state.highest_roe_reached = current_roe
        
        # ==================== SIGNAL REVALIDATION ====================
        if (self.config.exit_on_invalid_signal and 
            self.strategy and 
            candles and
            current_roe < 0):  # Only check when in loss
            
            should_check = (
                state.last_signal_check is None or
                (now - state.last_signal_check).total_seconds() >= self.config.revalidate_interval_sec
            )
            
            if should_check:
                state.last_signal_check = now
                
                # Revalidate signal
                try:
                    signal = await self._revalidate_signal(symbol, state, candles)
                    if signal:
                        is_still_valid = self._is_signal_still_valid(state, signal)
                        state.last_signal_valid = is_still_valid
                        
                        if not is_still_valid:
                            logger.warning(f"⚠️ {symbol} signal invalidated! Recommending exit.")
                            result['actions'].append('signal_invalid')
                            result['recommend_exit'] = True
                            return result
                except Exception as e:
                    logger.debug(f"Signal revalidation error: {e}")
        
        # ==================== TRAILING SL LOGIC ====================
        # Check if we should update SL based on current ROE
        trigger_crossed, new_sl_roe = self.get_next_sl_level(current_roe)
        
        # Only update if:
        # 1. We crossed a trigger level
        # 2. New SL is better (higher) than current SL
        # 3. We haven't updated too recently
        should_update_sl = (
            new_sl_roe is not None and
            new_sl_roe > state.current_sl_roe and
            (state.last_sl_update is None or 
             (now - state.last_sl_update).total_seconds() >= self.config.min_update_interval_sec)
        )
        
        if should_update_sl:
            # Calculate new SL price
            new_sl_price = self.get_sl_price_for_roe(state, new_sl_roe)
            
            logger.info(f"🔒 TRAILING TRIGGER: {symbol} at {current_roe:.1f}% ROE")
            logger.info(f"   Moving SL from {state.current_sl_roe}% to {new_sl_roe}% ROE")
            logger.info(f"   SL Price: ${new_sl_price:.4f}")
            
            try:
                # Cancel old SL first
                cancel_result = self.order_manager.cancel_sl_only(symbol, state.is_long)
                logger.debug(f"   Cancelled old SL: {cancel_result}")
                
                # Small delay to ensure cancellation propagates
                import asyncio
                await asyncio.sleep(0.3)
                
                # Place new SL
                update_result = await self.order_manager.modify_stops(
                    symbol=symbol,
                    new_sl=new_sl_price,
                    new_tp=None  # Don't touch TP
                )
                
                if update_result.get('success'):
                    state.current_sl_roe = new_sl_roe
                    state.last_sl_update = now
                    state.sl_order_placed = True
                    
                    result['actions'].append(f'sl_updated_to_{new_sl_roe}%')
                    
                    if new_sl_roe == 0:
                        logger.info(f"✅ {symbol}: SL moved to BREAKEVEN!")
                    elif new_sl_roe > 0:
                        logger.info(f"✅ {symbol}: Profit LOCKED at +{new_sl_roe}% ROE!")
                    else:
                        logger.info(f"✅ {symbol}: SL updated to {new_sl_roe}% ROE")
                else:
                    logger.warning(f"⚠️ Failed to update SL: {update_result.get('error')}")
                    result['actions'].append('sl_update_failed')
                    
            except Exception as e:
                logger.error(f"❌ Error updating trailing SL: {e}")
                result['actions'].append(f'error:{str(e)}')
        
        return result
    
    async def _revalidate_signal(
        self,
        symbol: str,
        state: PositionState,
        candles: list
    ) -> Optional[Dict]:
        """
        Revalidate the entry signal using current market data.
        
        Args:
            symbol: Trading pair
            state: Position state
            candles: Recent candles
            
        Returns:
            Current signal dict or None
        """
        if not self.strategy:
            return None
        
        try:
            # Get current signal from strategy
            signal = await self.strategy.evaluate(candles)
            return signal
        except Exception as e:
            logger.debug(f"Signal revalidation failed: {e}")
            return None
    
    def _is_signal_still_valid(self, state: PositionState, signal: Dict) -> bool:
        """
        Check if the current signal still supports the position direction.
        
        A signal is invalid if:
        - We're LONG but signal is now SHORT (or below threshold)
        - We're SHORT but signal is now LONG (or below threshold)
        
        Args:
            state: Position state
            signal: Current signal from strategy
            
        Returns:
            True if signal still supports position, False otherwise
        """
        if not signal:
            return True  # No signal = stay in position
        
        signal_direction = signal.get('direction')
        signal_score = signal.get('score', 0)
        
        # If signal is opposite direction with decent strength, consider invalid
        if state.is_long and signal_direction == 'short' and signal_score >= 8:
            return False
        if not state.is_long and signal_direction == 'long' and signal_score >= 8:
            return False
        
        return True
    
    async def place_initial_orders(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        is_long: bool,
        leverage: float
    ) -> Dict[str, Any]:
        """
        Place initial TP and SL orders for a new position.
        
        Args:
            symbol: Trading pair
            entry_price: Entry price
            size: Position size
            is_long: True if long position
            leverage: Position leverage
            
        Returns:
            Dict with order placement results
        """
        # Register position if not already tracked
        if symbol not in self.positions:
            self.register_position(symbol, entry_price, size, is_long, leverage)
        
        state = self.positions[symbol]
        
        # Calculate TP and SL prices
        tp_price = self.get_tp_price_for_roe(state, state.current_tp_roe)
        sl_price = self.get_sl_price_for_roe(state, state.current_sl_roe)
        
        logger.info(f"📊 Setting TP/SL for {symbol}")
        logger.info(f"   Entry: ${entry_price:.4f}")
        logger.info(f"   TP: ${tp_price:.4f} (+{state.current_tp_roe}% ROE)")
        logger.info(f"   SL: ${sl_price:.4f} ({state.current_sl_roe}% ROE)")
        
        try:
            result = await self.order_manager.modify_stops(
                symbol=symbol,
                new_sl=sl_price,
                new_tp=tp_price
            )
            
            if result.get('success'):
                state.sl_order_placed = True
                state.tp_order_placed = True
                logger.info(f"✅ TP/SL orders placed for {symbol}")
            else:
                logger.warning(f"⚠️ Failed to place TP/SL: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error placing initial orders: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_position_state(self, symbol: str) -> Optional[PositionState]:
        """Get the current state of a tracked position."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all tracked positions."""
        return self.positions.copy()


# Factory function
def create_trailing_manager(
    order_manager,
    strategy=None,
    tp_roe: float = 20.0,
    initial_sl_roe: float = -8.0,
    step_triggers: Optional[Dict[float, float]] = None
) -> TrailingManager:
    """
    Create a TrailingManager with custom configuration.
    
    Args:
        order_manager: HLOrderManager instance
        strategy: Strategy for signal revalidation (optional)
        tp_roe: Take profit ROE % (default 20%)
        initial_sl_roe: Initial stop loss ROE % (default -8%)
        step_triggers: Dict mapping trigger ROE to new SL ROE
        
    Returns:
        Configured TrailingManager instance
    """
    if step_triggers is None:
        step_triggers = {
            4.0: 0.0,    # At +4% ROE -> breakeven
            8.0: 4.0,    # At +8% ROE -> +4%
            12.0: 8.0,   # At +12% ROE -> +8%
            16.0: 12.0,  # At +16% ROE -> +12%
        }
    
    config = TrailingConfig(
        tp_roe_pct=tp_roe,
        initial_sl_roe_pct=initial_sl_roe,
        step_triggers=step_triggers
    )
    
    return TrailingManager(order_manager, strategy, config)
