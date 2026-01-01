"""
Pytest configuration and fixtures for Hyperbot tests.
"""

import pytest
import os
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock
from typing import List, Dict, Any

# Set test environment
os.environ.setdefault('ACCOUNT_ADDRESS', '0xTEST')
os.environ.setdefault('API_SECRET', 'test_secret')
os.environ.setdefault('TESTNET', 'true')
os.environ.setdefault('SYMBOL', 'BTC')
os.environ.setdefault('MAX_LEVERAGE', '10')


@pytest.fixture
def sample_candles() -> List[Dict[str, Any]]:
    """Generate sample OHLCV candles for testing."""
    base_price = 100.0
    candles = []
    
    for i in range(200):
        # Create realistic price movement
        variation = (i % 20 - 10) * 0.1  # Oscillates between -1% and +1%
        open_price = base_price + variation
        high_price = open_price * 1.005
        low_price = open_price * 0.995
        close_price = open_price * (1 + (i % 3 - 1) * 0.002)
        volume = 1000 + (i % 10) * 100
        
        # Spread candles across hours/minutes properly
        hours = i // 60
        minutes = i % 60
        
        candles.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'timestamp': datetime(2026, 1, 1, hours, minutes, 0, tzinfo=timezone.utc).isoformat()
        })
        
        base_price = close_price
    
    return candles


@pytest.fixture
def trending_up_candles() -> List[Dict[str, Any]]:
    """Generate candles showing a clear uptrend."""
    candles = []
    price = 100.0
    
    for i in range(200):
        # Consistent uptrend with small pullbacks
        if i % 5 == 0:
            change = -0.002  # Small pullback
        else:
            change = 0.003  # Upward movement
        
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * 1.001
        low_price = min(open_price, close_price) * 0.999
        
        candles.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': 1000 + i * 10
        })
        
        price = close_price
    
    return candles


@pytest.fixture
def trending_down_candles() -> List[Dict[str, Any]]:
    """Generate candles showing a clear downtrend."""
    candles = []
    price = 100.0
    
    for i in range(200):
        # Consistent downtrend with small bounces
        if i % 5 == 0:
            change = 0.002  # Small bounce
        else:
            change = -0.003  # Downward movement
        
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * 1.001
        low_price = min(open_price, close_price) * 0.999
        
        candles.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': 1000 + i * 10
        })
        
        price = close_price
    
    return candles


@pytest.fixture
def ranging_candles() -> List[Dict[str, Any]]:
    """Generate candles showing a ranging/sideways market."""
    candles = []
    base_price = 100.0
    
    for i in range(200):
        # Oscillate around base price
        import math
        offset = math.sin(i * 0.1) * 2  # Oscillate ±2%
        
        open_price = base_price + offset
        close_price = base_price + offset + (0.5 if i % 2 == 0 else -0.5)
        high_price = max(open_price, close_price) + 0.2
        low_price = min(open_price, close_price) - 0.2
        
        candles.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': 1000
        })
    
    return candles


@pytest.fixture
def mock_hl_client():
    """Mock HyperLiquid client for testing."""
    client = MagicMock()
    client.get_positions = AsyncMock(return_value=[])
    client.get_account_state = AsyncMock(return_value={
        'account_value': 1000.0,
        'margin_used': 0.0,
        'positions': []
    })
    client.round_price = MagicMock(side_effect=lambda s, p: round(p, 2))
    client.round_size = MagicMock(side_effect=lambda s, sz: round(sz, 4))
    return client


@pytest.fixture
def mock_order_manager(mock_hl_client):
    """Mock order manager for testing."""
    manager = MagicMock()
    manager.client = mock_hl_client
    manager.position_orders = {}
    manager.open_order = AsyncMock(return_value={'success': True, 'order_id': 'test123'})
    manager.close_position = AsyncMock(return_value={'success': True})
    manager.modify_stops = AsyncMock(return_value={'success': True})
    return manager


@pytest.fixture
def sample_position():
    """Sample position data."""
    return {
        'symbol': 'BTC',
        'size': 0.1,
        'entry_price': 50000.0,
        'mark_price': 51000.0,
        'unrealized_pnl': 100.0,
        'leverage': 10,
        'side': 'long'
    }


@pytest.fixture
def sample_account_state():
    """Sample account state."""
    return {
        'account_value': 1000.0,
        'margin_used': 100.0,
        'available_balance': 900.0,
        'positions': []
    }
