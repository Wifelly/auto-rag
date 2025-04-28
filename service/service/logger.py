# --- logger.py ---

import logging
from pathlib import Path


class LoggerSetup:
    def __init__(self, name: str = "pipeline_logger", log_file: str = "logs/pipeline.log") -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


logger = LoggerSetup().get_logger()

__all__ = ["logger"]
