# Logging utilities with structured output and file rotation
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "event"):
            log_data["event"] = record.event
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
):
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(log_level)
    
    formatter = StructuredFormatter()
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_event(event: str, payload: Dict[str, Any], level: str = "INFO"):
    logger = logging.getLogger(__name__)
    log_method = getattr(logger, level.lower(), logger.info)
    extra = {
        "event": event,
        "metadata": payload,
    }
    log_method(f"Event: {event}", extra=extra)


def log_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    payload = {
        "metric_name": name,
        "value": value,
    }
    if tags:
        payload["tags"] = tags
    
    log_event("metric", payload)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    payload = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    if context:
        payload["context"] = context
    
    log_event("error", payload, level="ERROR")