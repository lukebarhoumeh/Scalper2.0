from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple


def setup_logging(logs_dir: str) -> Tuple[logging.Logger, logging.Logger]:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=fmt)

    app_logger = logging.getLogger("bot")

    file_handler = RotatingFileHandler(
        str(Path(logs_dir) / "bot.log"), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    app_logger.addHandler(file_handler)

    audit_logger = logging.getLogger("audit")
    audit_handler = RotatingFileHandler(
        str(Path(logs_dir) / "trades_audit.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    audit_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(audit_handler)

    return app_logger, audit_logger


