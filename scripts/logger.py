"""
Centralized Logging Framework for Smart Parking System
Provides structured logging with file rotation and different log levels.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import config

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colors for console output.
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logger(
    name: str = "SmartParking",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (defaults to config.LOG_FILE)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output with rotation
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Format strings
    detailed_format = '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    simple_format = '%(asctime)s | %(levelname)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console Handler (with colors)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console
        try:
            console_formatter = ColoredFormatter(simple_format, datefmt=date_format)
        except:
            # Fallback to regular formatter if colors not supported
            console_formatter = logging.Formatter(simple_format, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File Handler (with rotation)
    if file_output:
        log_file = log_file or config.LOG_FILE
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler - keeps last 5 files of 10MB each
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        
        # Use detailed format for file
        file_formatter = logging.Formatter(detailed_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Gets a child logger with the specified name.
    Inherits configuration from the root logger.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"SmartParking.{name}")

# Create the main system logger
main_logger = setup_logger(
    name="SmartParking",
    level=logging.INFO,
    console_output=True,
    file_output=True
)

# Convenience functions for quick logging
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message"""
    main_logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs) -> None:
    """Log info message"""
    main_logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message"""
    main_logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs) -> None:
    """Log error message"""
    main_logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message"""
    main_logger.critical(msg, *args, **kwargs)

def exception(msg: str, *args, **kwargs) -> None:
    """Log exception with traceback"""
    main_logger.exception(msg, *args, **kwargs)

# Test functionality
if __name__ == "__main__":
    # Create test logger
    test_logger = setup_logger("TestLogger", level=logging.DEBUG)
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    try:
        raise ValueError("Test exception")
    except Exception:
        test_logger.exception("An exception occurred")
    
    print(f"\n✅ Log file created at: {config.LOG_FILE}")
