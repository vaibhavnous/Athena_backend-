import json
import logging
import os
import sys
from datetime import datetime, timezone

from logtail import LogtailHandler


class AthenaJsonFormatter(logging.Formatter):
    """Enhanced JSON formatter that captures custom structured fields."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }

        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ConsoleContextFormatter(logging.Formatter):
    """Makes terminal logs readable by prefixing the LangGraph node name."""

    def format(self, record):
        node = getattr(record, "node", "SYSTEM")
        return f"{self.formatTime(record, self.datefmt)} | {record.levelname:^7} | [{node}] {record.getMessage()}"


def get_athena_logger():
    logger = logging.getLogger("athena")
    logger.propagate = False

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        logtail_token = os.environ.get("LOGTAIL_TOKEN", "zDgMieaAtegpukJDifiCiRkQ")
        if logtail_token:
            logtail_handler = LogtailHandler(source_token=logtail_token)
            logger.addHandler(logtail_handler)

        suppress_console = os.environ.get("ATHENA_SUPPRESS_CONSOLE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not suppress_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleContextFormatter())
            logger.addHandler(console_handler)

        json_handler = logging.FileHandler("pipeline_logs.json", encoding="utf-8")
        json_handler.setFormatter(AthenaJsonFormatter())
        logger.addHandler(json_handler)

    return logger


logger = get_athena_logger()
