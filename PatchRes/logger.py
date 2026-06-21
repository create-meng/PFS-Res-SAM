"""Global logging system for experiment scripts."""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = None
GLOBAL_LOGGER = None


def setup_global_logger(base_dir: str, script_name: str, log_level: int = logging.DEBUG) -> logging.Logger:
    """Set up a global logger with file and console handlers."""
    global LOG_DIR, GLOBAL_LOGGER

    LOG_DIR = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    prefix = script_name.replace("_", "")
    log_filename = f"{prefix}_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger(f"pfs_res_sam_{prefix}")
    logger.setLevel(log_level)
    logger.handlers.clear()

    fh = RotatingFileHandler(log_path, maxBytes=100*1024*1024, backupCount=10, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)

    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(script)s] | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')

    fh.setFormatter(file_formatter)
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    class ScriptFilter(logging.Filter):
        def filter(self, record):
            record.script = script_name
            return True

    logger.addFilter(ScriptFilter())
    GLOBAL_LOGGER = logger

    logger.info("=" * 80)
    logger.info(f"{script_name} started")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 80)

    return logger


def get_logger() -> Optional[logging.Logger]:
    """Get the global logger instance."""
    return GLOBAL_LOGGER


def log_config(config: dict, logger: Optional[logging.Logger] = None):
    """Log config parameters."""
    if logger is None:
        logger = get_logger()
    if logger is None:
        return
    logger.info("Config:")
    for key, value in sorted(config.items()):
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        elif isinstance(value, (list, tuple)) and len(value) > 10:
            logger.info(f"  {key}: [{len(value)} items]")
        else:
            logger.info(f"  {key}: {value}")


def log_section(title: str, logger: Optional[logging.Logger] = None):
    """Log a section header."""
    if logger is None:
        logger = get_logger()
    if logger is None:
        return
    logger.info("-" * 80)
    logger.info(title)
    logger.info("-" * 80)


def log_finish(script_name: str, logger: Optional[logging.Logger] = None):
    """Log script completion."""
    if logger is None:
        logger = get_logger()
    if logger is None:
        return
    logger.info("=" * 80)
    logger.info(f"{script_name} completed")
    logger.info("=" * 80)
    logger.info("")
