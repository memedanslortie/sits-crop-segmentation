import logging
import sys
from pathlib import Path


def setup_logger(rank: int, log_dir: Path | None = None, run_name: str = "run") -> logging.Logger:
    """
    Rank 0 logs to console + file. Other ranks log warnings/errors to file only.
    Returns a logger that can be used anywhere after import.
    """
    logger = logging.getLogger("vjepa_seg")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        # Console handler — INFO and above
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{run_name}_rank{rank}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Retrieve the logger from anywhere without passing it around."""
    return logging.getLogger("vjepa_seg")