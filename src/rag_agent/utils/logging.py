"""JSON logging configuration."""

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": int(time.time()),
        }
        if hasattr(record, "extra"):
            base.update(record.extra)  # type: ignore
        return json.dumps(base, ensure_ascii=False)


def setup_logger(name: str = "rag") -> logging.Logger:
    """Setup a JSON logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger