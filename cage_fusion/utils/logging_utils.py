import os
import logging
from rich.logging import RichHandler
from loguru import logger as loguru_logger

# Ensure the logs directory exists
LOG_DIR = os.getenv("CAGE_FUSION_LOG_DIR", "/logs")
os.makedirs(LOG_DIR, exist_ok=True)
 
LOG_FILE = os.path.join(LOG_DIR, "cage_fusion.log")

# === 1. RichHandler for console ===
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)

# Standard logger for your project
logger = logging.getLogger("cagefusion")

# === 2. Loguru for file logging (all logs) ===
loguru_logger.remove()  # Remove default Loguru handlers

# Log to file with rotation
loguru_logger.add(
    LOG_FILE,
    rotation="10 MB",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    enqueue=True
)

# === 3. Propagate ALL standard logging to Loguru (file) ===
class PropagateToLoguru(logging.Handler):
    def emit(self, record):
        # Use Loguru's level if available, else default to record.levelno
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.log(level, record.getMessage())

# Attach handler to the root logger
logging.getLogger().addHandler(PropagateToLoguru())

# Now, any call to logger.info(), logger.warning(), etc
# will print to console (Rich) AND log to file (Loguru)!
