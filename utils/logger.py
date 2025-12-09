"""
Logger utility for the sentiment analysis project
Provides consistent logging across all modules
"""

import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str = __name__):
    """
    Get configured logger instance
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()
    
    # Get configuration from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file_path = os.getenv('LOG_FILE_PATH', './logs/sentiment_analysis.log')
    enable_console = os.getenv('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true'
    enable_file = os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true'
    
    # Create logs directory if it doesn't exist
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File logging format (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Add console handler
    if enable_console:
        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # Add file handler
    if enable_file:
        logger.add(
            log_file_path,
            format=file_format,
            level=log_level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress rotated logs
            backtrace=True,
            diagnose=True
        )
    
    return logger


# Create a default logger instance
default_logger = get_logger()


if __name__ == "__main__":
    # Test the logger
    test_logger = get_logger(__name__)
    
    test_logger.info("This is an info message")
    test_logger.success("This is a success message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.debug("This is a debug message")
    
    print("\n Logger test completed!")
    print(f"Logs are saved to: {os.path.abspath(os.getenv('LOG_FILE_PATH', './logs/sentiment_analysis.log'))}")
