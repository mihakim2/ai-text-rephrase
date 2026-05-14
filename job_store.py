import json
import threading
import queue
import time
import uuid
from typing import Any, Optional


class JobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def create_quick(self, source_filename: str, sentences: list[str], config: dict) -> str:
        job_id = str(uuid.uuid4())
        doc = {
            "job_id": job_id,
            "mode": "quick",
            "status": "pending",
            "source_filename": source_filename,
            "created_at": time.time(),
            "voice_profile": None,
            "user_samples": None,
            "sentences": [
                {
                    "id": i,
                    "original": s,
                    "intent": None,
                    "rephrased_versions": [],
                    "final_text": s,
                    "current_score": None,
                    "iterations_used": 0,
                    "status": "pending",
                }
                for i, s in enumerate(sentences)
            ],
            "config": config,
            "progress": {"done": 0, "total": len(sentences)},
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = doc
            self._queues[job_id] = queue.Queue()
        return job_id

    # Backwards-compatible alias used by older callers
    create = create_quick

    def create_deep(
        self,
        source_filename: str,
        raw_text: str,
        blocks: list[dict],
        user_samples: list[str] | None,
        config: dict,
    ) -> str:
        job_id = str(uuid.uuid4())
        prose_total = sum(1 for b in blocks if b.get("type") == "prose")
        doc = {
            "job_id": job_id,
            "mode": "deep",
            "status": "pending",
            "source_filename": source_filename,
            "created_at": time.time(),
            "raw_text": raw_text,
            "voice_profile": None,
            "user_samples": user_samples or None,
            "blocks": blocks,
            "config": config,
            "progress": {"done": 0, "total": prose_total},
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = doc
            self._queues[job_id] = queue.Queue()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def emit(self, job_id: str, event: str, data: Any):
        with self._lock:
            q = self._queues.get(job_id)
        if q is not None:
            q.put({"event": event, "data": data})

    def stream(self, job_id: str, timeout: float = 30.0):
        with self._lock:
            q = self._queues.get(job_id)
        if q is None:
            return
        while True:
            try:
                msg = q.get(timeout=timeout)
            except queue.Empty:
                yield {"event": "heartbeat", "data": {"ts": time.time()}}
                continue
            yield msg
            if msg["event"] in ("done", "error"):
                break

    def final_text(self, job_id: str) -> str:
        with self._lock:
            doc = self._jobs.get(job_id)
        if not doc:
            return ""
        if doc.get("mode") == "deep":
            import structure
            return structure.reassemble(doc["blocks"])
        return " ".join(s["final_text"] for s in doc["sentences"])

    def to_json(self, job_id: str) -> str:
        with self._lock:
            doc = self._jobs.get(job_id)
        return json.dumps(doc, indent=2) if doc else "{}"


store = JobStore()
