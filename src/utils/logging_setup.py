"""Logging setup"""
import sys
from pathlib import Path
from loguru import logger
from src.utils.config import config

def setup_logging():
    """Configure logging"""
    # Remove default handler
    logger.remove()
    
    # Add stdout handler
    logger.add(
        sys.stdout,
        format=config.get('logging.format', '{time} | {level} | {message}'),
        level=config.get('logging.level', 'INFO')
    )
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Add file handler
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level="DEBUG"
    )
    
    return logger

# Setup logger
log = setup_logging()
