import logging
import sys


class ColorFormatter(logging.Formatter):
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    BASE = "%(asctime)s %(levelname)-5s %(message)s"

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.GREY)
        fmt = f"{color}%(asctime)s %(levelname)-5s{self.RESET} %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


def setup_logging(level=logging.INFO):
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)


def job_log(job_id: str, msg: str) -> str:
    return f"[job:{job_id[:8]}] {msg}"


class _SSELogHandler(logging.Handler):
    """Routes log records into the job's SSE event queue so the UI can show them."""

    def __init__(self, job_id: str, store):
        super().__init__()
        self.job_id = job_id
        self.store = store
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record):
        try:
            self.store.emit(
                self.job_id,
                "log",
                {
                    "level": record.levelname,
                    "msg": self.format(record),
                },
            )
        except Exception:
            pass


def setup_job_logger(job_id: str, run_dir, store) -> logging.Logger:
    """Per-job logger. Writes to:
      1. Console (via root logger propagation; uses ColorFormatter)
      2. run_dir/log.txt (plain text)
      3. SSE 'log' event queue (UI live tail)
    """
    jlog = logging.getLogger(f"job.{job_id}")
    jlog.setLevel(logging.INFO)
    jlog.propagate = True  # root logger handles console output
    # Clear any handlers from a previous instantiation (defensive)
    jlog.handlers.clear()

    fh = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    jlog.addHandler(fh)
    jlog.addHandler(_SSELogHandler(job_id, store))
    return jlog
