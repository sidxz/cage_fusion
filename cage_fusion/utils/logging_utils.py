# cage_fusion/utils/logging_utils.py

import logging
from rich.logging import RichHandler
from loguru import logger as loguru_logger

# === Configure Rich logging for console ===
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)

# Create a standard logger for internal use
logger = logging.getLogger("cagefusion")

# === Configure Loguru for file logging ===
loguru_logger.remove()  # Remove Loguru's default handlers

# File logging with rotation
loguru_logger.add(
    "logs/cagefusion.log",
    rotation="10 MB",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Optional: Forward loguru logs to standard logger
class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger = logging.getLogger(record.name)
        logger.handle(record)

loguru_logger.add(InterceptHandler(), level="INFO")
