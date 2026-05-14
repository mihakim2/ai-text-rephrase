"""Thin dispatcher. Routes a job to quick_mode (sentence pipeline) or deep_mode
(paragraph pipeline) based on doc['mode'].

The actual pipelines live in:
  - quick_mode.py   (sentence-level)
  - deep_mode.py    (block-level with triage + blind detector + samples)
"""
import logging

import quick_mode
import deep_mode
from job_store import store

log = logging.getLogger(__name__)

# Re-export sentence splitter for the upload route's quick-mode path.
split_sentences = quick_mode.split_sentences


def run_job(job_id: str):
    doc = store.get(job_id)
    if not doc:
        log.error(f"[job:{job_id[:8]}] job not found")
        return
    mode = doc.get("mode", "quick")
    if mode == "deep":
        deep_mode.run(job_id)
    else:
        quick_mode.run(job_id)
