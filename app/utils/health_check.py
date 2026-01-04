"""
Health Check Module for Hyperbot

Provides HTTP endpoint for monitoring bot health status.
Useful for PM2, Docker, and other process managers.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from aiohttp import web

logger = logging.getLogger(__name__)


class HealthCheck:
    """
    Simple HTTP health check server for process monitoring.
    
    Exposes:
    - GET /health - Basic liveness check
    - GET /status - Detailed status with metrics
    - GET /ready  - Readiness probe (trading ready)
    
    Usage:
        health = HealthCheck(bot_instance)
        await health.start(port=8080)
    """
    
    def __init__(self, bot=None, port: int = 8080):
        """
        Initialize health check server.
        
        Args:
            bot: Reference to HyperLiquidBot instance (optional)
            port: HTTP port to listen on (default: 8080)
        """
        self.bot = bot
        self.port = port
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Health metrics
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)
        self.error_count = 0
        self.last_signal_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.signals_generated = 0
        self.trades_executed = 0
        
        logger.info(f"🏥 Health check initialized on port {port}")
    
    async def _block_scanners(self, request: web.Request) -> web.Response:
        """Block malicious scanners trying to find config files."""
        # Don't log every scanner attempt - they're noise
        return web.Response(status=403, text="Forbidden")
    
    async def start(self, port: int = None) -> None:
        """Start the health check HTTP server."""
        if port:
            self.port = port
            
        self.app = web.Application()
        
        # Only allow specific health check routes
        self.app.router.add_get('/health', self._handle_health)
        self.app.router.add_get('/status', self._handle_status)
        self.app.router.add_get('/ready', self._handle_ready)
        self.app.router.add_get('/metrics', self._handle_metrics)
        
        # Block all other routes (security scanners)
        self.app.router.add_route('*', '/{path:.*}', self._block_scanners)
        
        # Disable access logging for scanner noise
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        
        logger.info(f"🏥 Health check server started on http://0.0.0.0:{self.port}")
    
    async def stop(self) -> None:
        """Stop the health check server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("🏥 Health check server stopped")
    
    def record_heartbeat(self) -> None:
        """Update the heartbeat timestamp (call periodically from main loop)."""
        self.last_heartbeat = datetime.now(timezone.utc)
    
    def record_signal(self) -> None:
        """Record that a signal was generated."""
        self.last_signal_time = datetime.now(timezone.utc)
        self.signals_generated += 1
    
    def record_trade(self) -> None:
        """Record that a trade was executed."""
        self.last_trade_time = datetime.now(timezone.utc)
        self.trades_executed += 1
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        Basic liveness probe.
        Returns 200 if bot is running, 503 if stale.
        """
        now = datetime.now(timezone.utc)
        heartbeat_age = (now - self.last_heartbeat).total_seconds()
        
        # Consider unhealthy if no heartbeat in 5 minutes
        is_healthy = heartbeat_age < 300
        
        response = {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'timestamp': now.isoformat(),
            'heartbeat_age_seconds': round(heartbeat_age, 1),
        }
        
        status_code = 200 if is_healthy else 503
        return web.json_response(response, status=status_code)
    
    async def _handle_ready(self, request: web.Request) -> web.Response:
        """
        Readiness probe - is the bot ready to trade?
        Returns 200 if ready, 503 if not.
        """
        is_ready = False
        reasons = []
        
        if self.bot:
            # Check if bot has required connections
            has_client = hasattr(self.bot, 'hl_client') and self.bot.hl_client is not None
            has_strategy = hasattr(self.bot, 'strategy') and self.bot.strategy is not None
            has_candles = hasattr(self.bot, '_candles_cache') and self.bot._candles_cache and len(self.bot._candles_cache) >= 100
            
            if not has_client:
                reasons.append('Missing Hyperliquid client')
            if not has_strategy:
                reasons.append('Missing trading strategy')
            if not has_candles:
                reasons.append('Insufficient market data')
            
            is_ready = has_client and has_strategy and has_candles
        else:
            reasons.append('Bot instance not registered')
        
        response = {
            'ready': is_ready,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        if not is_ready:
            response['reasons'] = reasons
        
        status_code = 200 if is_ready else 503
        return web.json_response(response, status=status_code)
    
    async def _handle_status(self, request: web.Request) -> web.Response:
        """
        Detailed status endpoint with bot metrics.
        """
        now = datetime.now(timezone.utc)
        uptime = (now - self.start_time).total_seconds()
        
        response = {
            'status': 'running',
            'timestamp': now.isoformat(),
            'uptime_seconds': round(uptime, 1),
            'uptime_human': self._format_uptime(uptime),
            'metrics': {
                'signals_generated': self.signals_generated,
                'trades_executed': self.trades_executed,
                'error_count': self.error_count,
                'last_signal': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None,
            }
        }
        
        # Add bot-specific info if available
        if self.bot:
            response['bot'] = {
                'symbol': getattr(self.bot, 'symbol', 'unknown'),
                'mode': 'multi-asset' if getattr(self.bot, 'multi_asset_mode', False) else 'single',
            }
            
            # Get position info if available
            if hasattr(self.bot, 'positions') and self.bot.positions:
                response['positions'] = []
                for symbol, pos in self.bot.positions.items():
                    if pos.get('size', 0) != 0:
                        response['positions'].append({
                            'symbol': symbol,
                            'size': pos.get('size', 0),
                            'side': pos.get('side', 'unknown'),
                            'entry_price': pos.get('entry_price', 0),
                            'unrealized_pnl': pos.get('unrealized_pnl', 0),
                        })
        
        return web.json_response(response)
    
    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """
        Prometheus-compatible metrics endpoint.
        """
        now = datetime.now(timezone.utc)
        uptime = (now - self.start_time).total_seconds()
        heartbeat_age = (now - self.last_heartbeat).total_seconds()
        
        # Prometheus format
        lines = [
            '# HELP hyperbot_up Bot is running (1) or not (0)',
            '# TYPE hyperbot_up gauge',
            f'hyperbot_up{{symbol="{getattr(self.bot, "symbol", "unknown") if self.bot else "unknown"}"}} 1',
            '',
            '# HELP hyperbot_uptime_seconds Seconds since bot started',
            '# TYPE hyperbot_uptime_seconds counter',
            f'hyperbot_uptime_seconds {uptime:.1f}',
            '',
            '# HELP hyperbot_heartbeat_age_seconds Seconds since last heartbeat',
            '# TYPE hyperbot_heartbeat_age_seconds gauge',
            f'hyperbot_heartbeat_age_seconds {heartbeat_age:.1f}',
            '',
            '# HELP hyperbot_signals_total Total signals generated',
            '# TYPE hyperbot_signals_total counter',
            f'hyperbot_signals_total {self.signals_generated}',
            '',
            '# HELP hyperbot_trades_total Total trades executed',
            '# TYPE hyperbot_trades_total counter',
            f'hyperbot_trades_total {self.trades_executed}',
            '',
            '# HELP hyperbot_errors_total Total errors encountered',
            '# TYPE hyperbot_errors_total counter',
            f'hyperbot_errors_total {self.error_count}',
        ]
        
        return web.Response(text='\n'.join(lines), content_type='text/plain')
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)
        
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        
        return ' '.join(parts)


# Convenience function
async def create_health_server(bot=None, port: int = 8080) -> HealthCheck:
    """
    Create and start a health check server.
    
    Args:
        bot: Reference to HyperLiquidBot instance
        port: HTTP port (default: 8080)
    
    Returns:
        Running HealthCheck instance
    """
    health = HealthCheck(bot, port)
    await health.start()
    return health
