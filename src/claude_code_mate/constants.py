"""Shared constants for Claude Code Mate."""

from pathlib import Path

# Work directory for storing all data for Claude Code Mate
WORK_DIR = Path.home() / ".claude-code-mate"
CONFIG_PATH = WORK_DIR / "config.yaml"


def ensure_work_dir() -> None:
    """Ensure work directory exists."""
    WORK_DIR.mkdir(exist_ok=True)
