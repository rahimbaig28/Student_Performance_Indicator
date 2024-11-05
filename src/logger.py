import logging
import os
from datetime import datetime

# Corrected datetime formatting
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Corrected directory creation logic
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Corrected log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configuring logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

