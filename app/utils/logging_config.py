"""
Centralized Logging Configuration for Hyperbot

Provides structured logging with:
- Console output with colors
- File logging with rotation
- JSON format for production/analysis
- Different log levels per component
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import colorama for Windows color support
try:
    import colorama
    colorama.init()
except ImportError:
    pass


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colored output for console.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original level name
        record.levelname = levelname
        return result


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Useful for log aggregation and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'symbol'):
            log_data['symbol'] = record.symbol
        if hasattr(record, 'trade_id'):
            log_data['trade_id'] = record.trade_id
        
        return json.dumps(log_data)


def setup_logging(
    level: str = 'INFO',
    log_dir: str = 'logs',
    app_name: str = 'hyperbot',
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    json_logs: bool = False,
    console: bool = True,
) -> logging.Logger:
    """
    Configure centralized logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        app_name: Application name for log file naming
        max_file_size_mb: Max size per log file before rotation
        backup_count: Number of backup files to keep
        json_logs: Use JSON format for file logs
        console: Enable console output
    
    Returns:
        Root logger configured with handlers
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        console_format = '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s'
        console_handler.setFormatter(ColoredFormatter(console_format, datefmt='%H:%M:%S'))
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = log_path / f'{app_name}.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    
    if json_logs:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_format = '%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s'
        file_handler.setFormatter(logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S'))
    
    root_logger.addHandler(file_handler)
    
    # Error file handler (errors only)
    error_file = log_path / f'{app_name}_errors.log'
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8',
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s\n%(exc_info)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    root_logger.info(f"📋 Logging configured: level={level}, dir={log_dir}, json={json_logs}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Component-specific log levels (can be customized)
COMPONENT_LEVELS = {
    'app.bot': 'INFO',
    'app.hl.hl_client': 'INFO',
    'app.hl.hl_order_manager': 'INFO',
    'app.hl.hl_websocket': 'WARNING',  # Reduce WS noise
    'app.strategies': 'INFO',
    'app.risk': 'INFO',
    'app.utils.indicator_calculator': 'WARNING',  # Reduce indicator noise
}


def apply_component_levels():
    """Apply custom log levels to specific components."""
    for component, level in COMPONENT_LEVELS.items():
        logging.getLogger(component).setLevel(getattr(logging, level))


# Auto-configure on import if running as main app
if os.getenv('HYPERBOT_LOGGING', 'false').lower() == 'true':
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_dir = os.getenv('LOG_DIR', 'logs')
    json_logs = os.getenv('JSON_LOGS', 'false').lower() == 'true'
    
    setup_logging(
        level=log_level,
        log_dir=log_dir,
        json_logs=json_logs,
    )
    apply_component_levels()
