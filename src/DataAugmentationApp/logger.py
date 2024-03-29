## initializing logging in here for simply importing the logging file.
import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_directory = "logs"
log_file = f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}.log"

log_filepath = os.path.join(log_directory, log_file)
os.makedirs(log_directory, exist_ok = True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("DataAug")
