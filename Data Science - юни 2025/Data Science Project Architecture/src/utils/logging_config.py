# src/utils/logging_config.py
"""
Logging configuration for the asthma prediction project.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 log_dir: str = 'logs') -> None:
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            }
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        file_path = log_path / log_file
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(file_path),
            'mode': 'a'
        }
        config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)