"""Quick mode: sentence-level pipeline.

split sentences -> voice profile -> intent per sentence -> rephrase per sentence
-> critic per batch of N -> iterate -> final coherence pass.

Lower latency, smaller blast radius per call. For paragraph-cohesive rewrites
with structure preservation use deep_mode.
"""
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pysbd

import config
import llm_client
import run_dir as rd
import usage
from job_store import store
from logging_config import setup_job_logger

log = logging.getLogger(__name__)

_segmenter = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    raw = _segmenter.segment(text)
    return [s.strip() for s in raw if s and s.strip()]


def is_trivial(sentence: str) -> bool:
    """Skip headers, fragments, and very short sentences."""
    words = sentence.split()
    if len(words) < 8:
        return True
    if not sentence.rstrip().endswith((".", "!", "?", ";", ":")):
        return True
    return False


def _context(sentences: list[dict], idx: int) -> tuple[list[str], list[str]]:
    before = [s["original"] for s in sentences[max(0, idx - config.CONTEXT_BEFORE) : idx]]
    after = [s["original"] for s in sentences[idx + 1 : idx + 1 + config.CONTEXT_AFTER]]
    return before, after


def _emit_progress(job_id: str, sentence_idx: int, total: int, **extra):
    payload = {"sentence_idx": sentence_idx, "total": total, **extra}
    store.emit(job_id, "progress", payload)


def _current_text(doc: dict) -> str:
    """Concatenate the best-known text for every sentence (final, latest, or original)."""
    parts = []
    for s in doc["sentences"]:
        if s.get("final_text"):
            parts.append(s["final_text"])
        elif s.get("rephrased_versions"):
            parts.append(s["rephrased_versions"][-1]["text"])
        else:
            parts.append(s["original"])
    return " ".join(parts)


def _write_checkpoint(run_path: Path, doc: dict, jlog: logging.Logger):
    """Overwrite checkpoint.txt and doc.json with the current best draft."""
    rd.save_text(run_path, "checkpoint.txt", _current_text(doc))
    rd.save_json(run_path, "doc.json", doc)
    jlog.info(f"checkpoint saved -> {run_path / 'checkpoint.txt'}")


def _process_intent_batch(job_id: str, batch_idxs: list[int], doc: dict, jlog: logging.Logger):
    """Run intent extraction for a batch of sentences in parallel."""
    sentences = doc["sentences"]

    def worker(idx: int):
        if is_trivial(sentences[idx]["original"]):
            return idx, None
        before, after = _context(sentences, idx)
        intent = llm_client.extract_intent(sentences[idx]["original"], before, after, job_id=job_id)
        return idx, intent

    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in batch_idxs]
        for fut in as_completed(futures):
            idx, intent = fut.result()
            if intent:
                sentences[idx]["intent"] = intent


def _rephrase_batch(
    job_id: str,
    batch_idxs: list[int],
    doc: dict,
    iteration: int,
    jlog: logging.Logger,
    fix_hints: Optional[dict[int, str]] = None,
):
    sentences = doc["sentences"]
    voice_profile = doc["voice_profile"]
    total = len(sentences)
    fix_hints = fix_hints or {}

    def worker(idx: int):
        s = sentences[idx]
        if is_trivial(s["original"]):
            s["final_text"] = s["original"]
            s["status"] = "skipped"
            return idx, s["original"]
        if not s["intent"]:
            s["final_text"] = s["original"]
            s["status"] = "skipped"
            return idx, s["original"]
        before, after = _context(sentences, idx)
        target = s["original"]
        if iteration > 1 and s["rephrased_versions"]:
            target = s["rephrased_versions"][-1]["text"]
        new_text = llm_client.rephrase(
            target=target,
            before=before,
            after=after,
            intent=s["intent"],
            voice_profile=voice_profile,
            fix_hint=fix_hints.get(idx),
            job_id=job_id,
        )
        return idx, new_text

    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in batch_idxs]
        for fut in as_completed(futures):
            idx, new_text = fut.result()
            s = sentences[idx]
            if s["status"] != "skipped":
                s["rephrased_versions"].append(
                    {"iteration": iteration, "text": new_text, "critic_score": None, "critic_notes": None}
                )
                s["iterations_used"] = iteration
                s["status"] = "rephrased"
            _emit_progress(
                job_id,
                sentence_idx=idx,
                total=total,
                iteration=iteration,
                status=s["status"],
                preview=s["rephrased_versions"][-1]["text"][:120] if s["rephrased_versions"] else s["original"][:120],
            )


def _critic_pass(
    job_id: str,
    batch_idxs: list[int],
    doc: dict,
    iteration: int,
    jlog: logging.Logger,
) -> dict:
    sentences = doc["sentences"]
    active = [
        {"idx": i, "text": sentences[i]["rephrased_versions"][-1]["text"]}
        for i in batch_idxs
        if sentences[i]["status"] not in ("skipped",) and sentences[i]["rephrased_versions"]
    ]
    if not active:
        return {"ai_likelihood_score": 10, "batch_flow_score": 10, "per_sentence": []}
    result = llm_client.critique_batch(active, job_id=job_id)
    for per in result.get("per_sentence", []):
        i = per.get("idx")
        if i is None or i not in batch_idxs:
            continue
        s = sentences[i]
        if s["rephrased_versions"]:
            s["rephrased_versions"][-1]["critic_score"] = per.get("score")
            s["rephrased_versions"][-1]["critic_notes"] = "; ".join(per.get("issues", []) or [])
            s["current_score"] = per.get("score")
    jlog.info(
        f"critic iter={iteration} batch={batch_idxs[0]}-{batch_idxs[-1]} "
        f"ai={result.get('ai_likelihood_score')} flow={result.get('batch_flow_score')} "
        f"tells={result.get('top_tells')}"
    )
    return result


def _pick_best_versions(batch_idxs: list[int], doc: dict):
    sentences = doc["sentences"]
    for i in batch_idxs:
        s = sentences[i]
        if s["status"] == "skipped":
            continue
        if not s["rephrased_versions"]:
            s["final_text"] = s["original"]
            continue
        scored = [v for v in s["rephrased_versions"] if v.get("critic_score") is not None]
        if scored:
            best = max(scored, key=lambda v: v["critic_score"])
        else:
            best = s["rephrased_versions"][-1]
        s["final_text"] = best["text"]
        s["current_score"] = best.get("critic_score")
        s["status"] = "accepted"


def _process_batch(job_id: str, batch_idxs: list[int], doc: dict, jlog: logging.Logger, *, max_iter: int, target_ai: int, target_flow: int):
    sentences = doc["sentences"]
    jlog.info(f"batch start: sentences {batch_idxs[0]}..{batch_idxs[-1]}")

    _rephrase_batch(job_id, batch_idxs, doc, iteration=1, jlog=jlog)
    critic = _critic_pass(job_id, batch_idxs, doc, iteration=1, jlog=jlog)

    iteration = 1
    prev_overall = None
    while iteration < max_iter:
        overall = critic.get("ai_likelihood_score", 0)
        flow = critic.get("batch_flow_score", 0)
        if overall >= target_ai and flow >= target_flow:
            jlog.info(
                f"batch {batch_idxs[0]}..{batch_idxs[-1]} passed at iter={iteration} (ai={overall} flow={flow})"
            )
            break
        fix_hints: dict[int, str] = {}
        for per in critic.get("per_sentence", []):
            i = per.get("idx")
            if i in batch_idxs and per.get("score", 10) < target_ai:
                fix_hints[i] = per.get("suggested_fix") or "Rewrite in a more natural human voice."
        if not fix_hints:
            break
        if prev_overall is not None and overall <= prev_overall:
            jlog.info(
                f"batch {batch_idxs[0]}..{batch_idxs[-1]} no improvement (prev={prev_overall} now={overall}), accepting best"
            )
            break
        prev_overall = overall
        iteration += 1
        jlog.info(
            f"batch {batch_idxs[0]}..{batch_idxs[-1]} retrying iter={iteration} with {len(fix_hints)} fixes"
        )
        _rephrase_batch(
            job_id, list(fix_hints.keys()), doc, iteration=iteration, jlog=jlog, fix_hints=fix_hints
        )
        critic = _critic_pass(job_id, batch_idxs, doc, iteration=iteration, jlog=jlog)

    _pick_best_versions(batch_idxs, doc)
    total = len(sentences)
    for i in batch_idxs:
        s = sentences[i]
        _emit_progress(
            job_id,
            sentence_idx=i,
            total=total,
            iteration=s["iterations_used"],
            status=s["status"],
            score=s.get("current_score"),
            preview=s["final_text"][:120],
        )
    doc["progress"]["done"] = max(doc["progress"]["done"], batch_idxs[-1] + 1)


def run(job_id: str):
    t0 = time.time()
    doc = store.get(job_id)
    if not doc:
        log.error(f"[job:{job_id[:8]}] job not found")
        return

    run_path = rd.make_run_dir()
    doc["run_dir"] = str(run_path)
    store.update(job_id, run_dir=str(run_path))

    jlog = setup_job_logger(job_id, run_path, store)
    jlog.info(f"run dir: {run_path}")

    # Resolve per-job thresholds (intensity preset, set at upload time).
    T = (doc.get("config") or {}).get("thresholds") or {}
    max_iter = T.get("MAX_ITERATIONS", config.MAX_ITERATIONS)
    target_ai = T.get("TARGET_AI_SCORE", config.TARGET_AI_SCORE)
    target_flow = T.get("TARGET_FLOW_SCORE", config.TARGET_FLOW_SCORE)
    jlog.info(
        f"intensity: {doc.get('config', {}).get('intensity', 'default')} "
        f"(target_ai>={target_ai}, target_flow>={target_flow}, max_iter={max_iter})"
    )

    try:
        store.update(job_id, status="running")
        sentences = doc["sentences"]
        total = len(sentences)
        jlog.info(f"start: {total} sentences from '{doc['source_filename']}'")

        # Save original input as best-reconstruction (joined sentences)
        rd.save_text(run_path, "input.txt", " ".join(s["original"] for s in sentences))

        # 1. Voice calibration
        sample_pool = [s["original"] for s in sentences if not is_trivial(s["original"])]
        if len(sample_pool) >= config.VOICE_SAMPLE_SIZE:
            sample = random.sample(sample_pool, config.VOICE_SAMPLE_SIZE)
        else:
            sample = sample_pool or [s["original"] for s in sentences]
        jlog.info("voice calibration...")
        doc["voice_profile"] = llm_client.extract_voice_profile(sample, job_id=job_id)
        rd.save_json(run_path, "voice_profile.json", doc["voice_profile"])
        store.emit(job_id, "progress", {"stage": "voice_profile", "voice_profile": doc["voice_profile"]})
        jlog.info(
            f"voice: field={doc['voice_profile'].get('field_guess')} "
            f"tone={doc['voice_profile'].get('tone_notes')}"
        )

        # 2. Intent extraction
        jlog.info("extracting intents...")
        all_idxs = list(range(total))
        chunk_size = config.PARALLEL_WORKERS * 2
        for i in range(0, len(all_idxs), chunk_size):
            chunk = all_idxs[i : i + chunk_size]
            _process_intent_batch(job_id, chunk, doc, jlog)
            store.emit(job_id, "progress", {"stage": "intent", "done": i + len(chunk), "total": total})
            jlog.info(f"intents: {i + len(chunk)}/{total}")

        # 3. Rephrase in batches, with checkpoints
        processed_since_checkpoint = 0
        for start in range(0, total, config.BATCH_SIZE):
            batch_idxs = list(range(start, min(start + config.BATCH_SIZE, total)))
            _process_batch(job_id, batch_idxs, doc, jlog, max_iter=max_iter, target_ai=target_ai, target_flow=target_flow)
            processed_since_checkpoint += len(batch_idxs)
            if processed_since_checkpoint >= config.CHECKPOINT_EVERY:
                _write_checkpoint(run_path, doc, jlog)
                processed_since_checkpoint = 0

        # Always checkpoint before the final pass (in case it fails)
        _write_checkpoint(run_path, doc, jlog)

        # 4. Final coherence pass
        jlog.info("final coherence pass...")
        store.emit(job_id, "progress", {"stage": "final_coherence"})
        joined = " ".join(s["final_text"] for s in sentences)
        final_doc = llm_client.final_coherence_pass(joined, doc["voice_profile"], job_id=job_id)
        doc["final_output"] = final_doc
        rd.save_text(run_path, "output.txt", final_doc)
        rd.save_json(run_path, "doc.json", doc)

        duration = time.time() - t0
        accepted = [s for s in sentences if s["status"] == "accepted"]
        avg = (
            sum(s.get("current_score") or 0 for s in accepted) / max(1, len(accepted))
        )
        usage_summary = usage.tracker.summary(job_id)
        rd.save_json(run_path, "usage.json", usage_summary)
        store.update(job_id, status="done")
        store.emit(
            job_id,
            "done",
            {
                "duration_sec": round(duration, 1),
                "sentences": total,
                "avg_score": round(avg, 2),
                "run_dir": str(run_path),
                "usage": usage_summary,
            },
        )
        jlog.info(f"complete sentences={total} avg_score={avg:.2f} duration={duration:.1f}s")
        jlog.info(
            f"tokens: calls={usage_summary['totals']['calls']} "
            f"input={usage_summary['totals']['input']:,} "
            f"output={usage_summary['totals']['output']:,} "
            f"total={usage_summary['totals']['total']:,}"
        )
        for model, m in usage_summary["per_model"].items():
            jlog.info(f"  {model}: {m['calls']} calls, {m['total']:,} tokens (in={m['input']:,} out={m['output']:,})")
        jlog.info(f"output -> {run_path / 'output.txt'}")
    except Exception as e:
        log.exception(f"[job:{job_id[:8]}] error: {e}")
        try:
            jlog.error(f"error: {e}")
            # Always checkpoint on failure so user can still recover something
            _write_checkpoint(run_path, doc, jlog)
            jlog.info("checkpoint saved on error — partial output recoverable in checkpoint.txt")
        except Exception:
            pass
        store.update(job_id, status="error", error=str(e))
        store.emit(job_id, "error", {"message": str(e)})
