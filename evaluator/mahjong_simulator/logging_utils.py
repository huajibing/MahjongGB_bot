import logging
import os

LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "game_log.txt")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a logger
logger = logging.getLogger("mahjong_simulator")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create a file handler
file_handler = logging.FileHandler(LOG_FILE, mode='w') # 'w' to overwrite log file for each run
file_handler.setLevel(logging.DEBUG) # Also set the handler to DEBUG

# Create a console handler for high-level feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Only INFO and above for console

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent logging from propagating to the root logger
logger.propagate = False
