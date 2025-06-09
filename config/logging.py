import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_dir: str = "data/logs") -> None:
    """Configure application logging with file and console handlers."""
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = os.path.join(log_dir, f"rag_agent_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5  # 10MB files, keep 5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)