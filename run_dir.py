"""Run-artifact directory layout.

runs/
  2026-05-14/         <- date
    13-45-22/         <- time (one folder per job)
      input.txt
      voice_profile.json
      log.txt
      checkpoint.txt   (overwritten every CHECKPOINT_EVERY sentences)
      output.txt       (final)
      doc.json         (full state, for debugging)
"""
import json
from datetime import datetime
from pathlib import Path

import config


def make_run_dir() -> Path:
    now = datetime.now()
    base = Path(config.RUNS_DIR) / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_text(run_dir: Path, name: str, text: str) -> Path:
    p = run_dir / name
    p.write_text(text, encoding="utf-8")
    return p


def save_json(run_dir: Path, name: str, data) -> Path:
    p = run_dir / name
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
