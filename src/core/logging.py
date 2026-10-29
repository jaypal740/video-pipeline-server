import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
# Libraries that log every request or query; their chatter hides the pipeline's own lines.
QUIET = ("httpx", "httpcore", "openai", "anthropic", "uvicorn.access", "sqlalchemy")
FILE_BYTES = 5 * 1024 * 1024
FILE_COUNT = 5


def configure_logging(level: str, log_dir: str | None) -> None:
    """Pipeline logs go to the console and, when log_dir is set, to a rotating file in it. Safe to call
    more than once: handlers are replaced, not added."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file = logging.handlers.RotatingFileHandler(path / "pipeline.log", maxBytes=FILE_BYTES,
                                                    backupCount=FILE_COUNT, encoding="utf-8")
        file.setFormatter(formatter)
        root.addHandler(file)
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
