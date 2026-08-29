import logging
import sys
from typing import Optional


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures structured logging for ResearchMind application backend.
    """
    logger = logging.getLogger("researchmind")
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
