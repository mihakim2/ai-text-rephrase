"""Per-job, per-model token usage tracking.

Every llm_client call records its API response's token counts here, keyed by job.
At end-of-run we get a summary by model (calls, input, output, cache, total).
"""
from __future__ import annotations

import threading
from typing import Any


def _zero_entry() -> dict[str, int]:
    return {
        "calls": 0,
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_create": 0,
        "total": 0,
    }


class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        # job_id -> {model_name -> usage_dict}
        self._usage: dict[str, dict[str, dict[str, int]]] = {}
        # job_id -> {role -> model_name}  (e.g. {"rephraser": "claude-sonnet-4-6"})
        # Useful so the UI can label which model played which role.
        self._roles: dict[str, dict[str, str]] = {}

    def record(
        self,
        job_id: str | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_create: int = 0,
        role: str | None = None,
    ):
        if not job_id:
            return
        with self._lock:
            jobs = self._usage.setdefault(job_id, {})
            entry = jobs.setdefault(model, _zero_entry())
            entry["calls"] += 1
            entry["input"] += int(input_tokens or 0)
            entry["output"] += int(output_tokens or 0)
            entry["cache_read"] += int(cache_read or 0)
            entry["cache_create"] += int(cache_create or 0)
            entry["total"] = entry["input"] + entry["output"]
            if role:
                self._roles.setdefault(job_id, {})[role] = model

    def summary(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            per_model = {m: dict(v) for m, v in self._usage.get(job_id, {}).items()}
            roles = dict(self._roles.get(job_id, {}))
        totals = _zero_entry()
        for v in per_model.values():
            for k in totals:
                totals[k] += v[k]
        return {"per_model": per_model, "roles": roles, "totals": totals}

    def clear(self, job_id: str):
        with self._lock:
            self._usage.pop(job_id, None)
            self._roles.pop(job_id, None)


tracker = UsageTracker()


# ---- Helpers used by llm_client to record SDK responses uniformly ----

def record_anthropic(job_id: str | None, model: str, resp, role: str | None = None):
    if not job_id:
        return
    u = getattr(resp, "usage", None)
    if not u:
        return
    tracker.record(
        job_id,
        model,
        getattr(u, "input_tokens", 0) or 0,
        getattr(u, "output_tokens", 0) or 0,
        getattr(u, "cache_read_input_tokens", 0) or 0,
        getattr(u, "cache_creation_input_tokens", 0) or 0,
        role=role,
    )


def record_gemini(job_id: str | None, model: str, resp, role: str | None = None):
    if not job_id:
        return
    um = getattr(resp, "usage_metadata", None)
    if not um:
        return
    tracker.record(
        job_id,
        model,
        getattr(um, "prompt_token_count", 0) or 0,
        getattr(um, "candidates_token_count", 0) or 0,
        getattr(um, "cached_content_token_count", 0) or 0,
        0,
        role=role,
    )
