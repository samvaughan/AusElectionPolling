import logging
import sys
from datetime import datetime
import os


class Logger:
    """
    A logger class that outputs logs to both stdout and a file.

    - INFO and higher messages are printed to stdout.
    - DEBUG and higher messages are saved to the log file.
    """

    def __init__(self, name=None, log_dir="logs", level=logging.DEBUG):
        """
        Initializes and configures the logger.

        :param name: Name of the logger.
        :param log_dir: Directory to save log files.
        :param level: Logging level.
        """

        if name is None:
            name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers if the logger is already configured
        if self.logger.hasHandlers():
            return

        # Create log file with current date
        log_file = (
            f"{log_dir}/AusElectionPolls_{datetime.now().strftime('%Y-%m-%d')}.log"
        )

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File Handler (DEBUG and higher)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Stream Handler (INFO and higher)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Add Handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
