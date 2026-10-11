import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Constants for log configuration
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}.log"  # one file per day
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3                # number of old log files to keep

# Project root = parent of the src/ folder (this file is src/logger.py)
root_dir = Path(__file__).resolve().parents[1]
log_dir_path = root_dir / LOG_DIR
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE

def configure_logger():
    """
    Configure logging with a rotating file handler and a console handler.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Do nothing if handler already exist (prevents duplicate lines)
    if logger.handlers:
        return

    # Define formatter
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

    # File handler with rotation: keeps everything, including DEBUG
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler: only INFO and above, to keep the terminal readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Quiet noisy third-party libraries so they don't flood the DEBUG log
    for noisy in ["maptlotlib", "numba", "PIL", "urllib3", "shap"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

# Configure the logger once, when the module is first imported
configure_logger()
