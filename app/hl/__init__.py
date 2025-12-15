"""
HyperLiquid Integration Module

Ultra-lean SDK integration with direct method passthrough.
Total: ~580 lines (vs 1,970 lines legacy = 71% reduction)

Components:
- HyperLiquidClient: Direct SDK passthrough for Exchange & Info
- HLOrderManager: Atomic TP/SL via bulk_orders(grouping="normalTpsl")
- HLWebSocket: SDK native subscriptions (userFills, orderUpdates, etc.)
"""

from app.hl.hl_client import HyperLiquidClient
from app.hl.hl_order_manager import HLOrderManager
from app.hl.hl_websocket import HLWebSocket

__all__ = [
    'HyperLiquidClient',
    'HLOrderManager', 
    'HLWebSocket',
]
