import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Add extra fields if available
        if hasattr(record, "props"):
            log_obj.update(record.props)

        return json.dumps(log_obj)


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Prevent duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
