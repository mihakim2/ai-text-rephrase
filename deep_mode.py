"""Deep mode: paragraph-level pipeline with structure preservation.

Pipeline:
  1. Parse the document into typed blocks (prose / header / list / code / math / blank).
     Only prose blocks are candidates for rephrasing — everything else passes through verbatim.
  2. Voice calibration:
       - If the user provided writing samples, infer voice directly from those.
       - Otherwise sample paragraphs from the document.
  3. TRIAGE: for each prose block, run the BLIND DETECTOR (Gemini, no rubric) on the original.
       - Score >= DEEP_PASSTHROUGH_SCORE → passthrough (preserve original, don't touch).
       - Below threshold → mark for rewriting.
  4. Intent extraction (Haiku, per flagged block, in parallel).
  5. Rephrase flagged blocks in batches:
       a. Rephrase the chunk (Sonnet) with paragraph context, intent, voice, samples.
       b. PROMPTED CRITIC scores each chunk (Gemini, with rubric, suggests fixes).
       c. BLIND DETECTOR scores each chunk independently (Gemini, no rubric).
       d. Accept when BOTH critic AND detector pass thresholds.
       e. Otherwise iterate up to DEEP_MAX_ITERATIONS with critic's fix hint.
       f. Always accept best-by-detector-score version.
  6. Optional final coherence pass over the full reassembled doc.
  7. Reassemble blocks preserving original paragraph breaks / headers / lists / code / math.

Why this beats sentence-level (quick mode):
  - Structure preserved: headers/code/math/lists never touched.
  - Triage skips blocks already human enough → fewer API calls, preserves good writing.
  - Cohesive paragraphs: writer sees the whole paragraph at once, not isolated sentences.
  - Dual-signal gate: prompted critic Goodharts against its own rubric; blind detector
    is the honest acceptance signal.
  - User samples (when provided) override the inferred voice — beats any auto-extracted profile.
"""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import config
import llm_client
import run_dir as rd
import structure
import usage
from job_store import store
from logging_config import setup_job_logger

log = logging.getLogger(__name__)


def _prose_context(blocks: list[dict], idx: int) -> tuple[Optional[str], Optional[str]]:
    """Return the immediately-preceding and following prose paragraphs (skipping non-prose)."""
    before = None
    for j in range(idx - 1, -1, -1):
        if blocks[j]["type"] == "prose":
            before = blocks[j]["original"]
            break
    after = None
    for j in range(idx + 1, len(blocks)):
        if blocks[j]["type"] == "prose":
            after = blocks[j]["original"]
            break
    return before, after


def _emit_block(job_id: str, blk: dict, total: int):
    store.emit(
        job_id,
        "progress",
        {
            "block_idx": blk["id"],
            "block_type": blk["type"],
            "total": total,
            "status": blk["status"],
            "detector_score": blk.get("detector_score"),
            "detector_original": (blk.get("detector_original") or {}).get("score"),
            "critic_score": blk.get("critic_score"),
            "similarity_score": blk.get("similarity_score"),
            "iteration": blk.get("iterations_used", 0),
            "preview": (blk.get("final_text") or blk["original"])[:140],
        },
    )


def _checkpoint(run_path: Path, doc: dict, jlog: logging.Logger):
    rd.save_text(run_path, "checkpoint.txt", structure.reassemble(doc["blocks"]))
    rd.save_json(run_path, "doc.json", doc)
    jlog.info(f"checkpoint saved -> {run_path / 'checkpoint.txt'}")


def _triage_prose_blocks(job_id: str, prose_idxs: list[int], blocks: list[dict], total: int, jlog: logging.Logger):
    """Run blind detector on each original prose block. Sets needs_rewrite + status."""
    def worker(idx: int):
        text = blocks[idx]["original"]
        try:
            result = llm_client.detect_ai(text, job_id=job_id)
        except Exception as e:
            jlog.warning(f"detector failed on block {idx}: {e}")
            result = {"score": 0, "reasoning": f"detector error: {e}"}
        return idx, result

    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in prose_idxs]
        for fut in as_completed(futures):
            idx, result = fut.result()
            b = blocks[idx]
            b["detector_original"] = result
            score = result.get("score", 0) or 0
            if score >= config.DEEP_PASSTHROUGH_SCORE:
                b["needs_rewrite"] = False
                b["status"] = "passthrough"
                b["final_text"] = b["original"]
            else:
                b["needs_rewrite"] = True
                b["status"] = "triage"
            _emit_block(job_id, b, total)
            jlog.info(
                f"triage block {idx}: detector={score} "
                f"-> {'rewrite' if b['needs_rewrite'] else 'passthrough'}"
            )


def _extract_intents(job_id: str, idxs: list[int], blocks: list[dict], jlog: logging.Logger):
    def worker(idx: int):
        before, after = _prose_context(blocks, idx)
        try:
            return idx, llm_client.extract_chunk_intent(blocks[idx]["original"], before, after, job_id=job_id)
        except Exception as e:
            jlog.warning(f"intent failed on block {idx}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in idxs]
        for fut in as_completed(futures):
            idx, intent = fut.result()
            blocks[idx]["intent"] = intent


def _rephrase_block_set(
    job_id: str,
    idxs: list[int],
    blocks: list[dict],
    voice_profile: dict,
    user_samples: Optional[list[str]],
    iteration: int,
    fix_hints: dict[int, str],
    total: int,
):
    def worker(idx: int):
        b = blocks[idx]
        if not b.get("intent"):
            return idx, None
        before, after = _prose_context(blocks, idx)
        target = b["original"]
        if iteration > 1 and b["rephrased_versions"]:
            target = b["rephrased_versions"][-1]["text"]
        try:
            text = llm_client.rephrase_chunk(
                paragraph=target,
                before_paragraph=before,
                after_paragraph=after,
                intent=b["intent"],
                voice_profile=voice_profile,
                user_samples=user_samples,
                fix_hint=fix_hints.get(idx),
                job_id=job_id,
            )
            return idx, text
        except Exception as e:
            return idx, e

    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in idxs]
        for fut in as_completed(futures):
            idx, text = fut.result()
            b = blocks[idx]
            if isinstance(text, Exception) or text is None:
                continue
            b["rephrased_versions"].append({"iteration": iteration, "text": text, "critic_score": None, "detector_score": None, "critic_notes": None})
            b["iterations_used"] = iteration
            b["status"] = "rephrased"
            _emit_block(job_id, b, total)


def _critic_and_detect(
    job_id: str,
    idxs: list[int],
    blocks: list[dict],
    iteration: int,
    jlog: logging.Logger,
    total: int,
) -> dict:
    """Run critic over a batch (one call) + blind detector per block in parallel."""
    chunks = [
        {
            "idx": i,
            "original": blocks[i]["original"],
            "rewrite": blocks[i]["rephrased_versions"][-1]["text"],
        }
        for i in idxs
        if blocks[i]["rephrased_versions"]
    ]
    if not chunks:
        return {"batch_flow_score": 10, "per_chunk": []}
    critic = llm_client.critique_chunks(chunks, job_id=job_id)
    # Apply critic scores (handle both new "ai_score"/"similarity_score" and legacy "score")
    for per in critic.get("per_chunk", []):
        i = per.get("idx")
        if i in idxs and blocks[i]["rephrased_versions"]:
            ai_s = per.get("ai_score", per.get("score"))
            sim_s = per.get("similarity_score")
            v = blocks[i]["rephrased_versions"][-1]
            v["critic_score"] = ai_s
            v["similarity_score"] = sim_s
            v["critic_notes"] = "; ".join(per.get("issues", []) or [])
            blocks[i]["critic_score"] = ai_s
            blocks[i]["similarity_score"] = sim_s
    # Parallel blind detector
    def detect_worker(idx):
        text = blocks[idx]["rephrased_versions"][-1]["text"]
        try:
            return idx, llm_client.detect_ai(text, job_id=job_id)
        except Exception as e:
            return idx, {"score": 0, "reasoning": f"detector error: {e}"}

    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as ex:
        futures = [ex.submit(detect_worker, i) for i in idxs]
        for fut in as_completed(futures):
            idx, result = fut.result()
            blocks[idx]["rephrased_versions"][-1]["detector_score"] = result.get("score")
            blocks[idx]["detector_score"] = result.get("score")
            _emit_block(job_id, blocks[idx], total)
    # Per-block summary in the log
    for i in idxs:
        b = blocks[i]
        if b.get("critic_score") is None:
            continue
        jlog.info(
            f"  block {i}: ai={b.get('critic_score')} sim={b.get('similarity_score')} det={b.get('detector_score')}"
        )
    jlog.info(
        f"iter={iteration} batch={idxs[0]}-{idxs[-1]} "
        f"flow={critic.get('batch_flow_score')} tells={critic.get('top_tells')}"
    )
    return critic


def _pick_best(idx: int, blocks: list[dict]):
    """For a block, pick the best rephrased version.

    Composite score: detector_score (honesty signal) penalized when similarity is too high.
    A version that hits the detector target but is near-copy of the original is *not* what we want.
    """
    b = blocks[idx]
    if not b["rephrased_versions"]:
        b["final_text"] = b["original"]
        return

    def score_of(v):
        det = v.get("detector_score")
        sim = v.get("similarity_score")
        if det is None:
            return -1
        # Penalize similarity above the max threshold; each point of excess subtracts from det.
        sim_penalty = max(0, (sim or 0) - config.DEEP_MAX_SIMILARITY_SCORE)
        return det - sim_penalty

    scored = [v for v in b["rephrased_versions"] if v.get("detector_score") is not None]
    if scored:
        best = max(scored, key=score_of)
    else:
        best = b["rephrased_versions"][-1]
    b["final_text"] = best["text"]
    b["detector_score"] = best.get("detector_score")
    b["critic_score"] = best.get("critic_score")
    b["similarity_score"] = best.get("similarity_score")
    b["status"] = "accepted"


def _process_chunk_batch(
    job_id: str,
    idxs: list[int],
    blocks: list[dict],
    voice_profile: dict,
    user_samples: Optional[list[str]],
    jlog: logging.Logger,
    total: int,
) -> bool:
    """Returns True if any block in this batch needed >1 iteration (for skipping the final pass)."""
    iteration = 1
    needed_iteration = False
    _rephrase_block_set(job_id, idxs, blocks, voice_profile, user_samples, iteration, {}, total)
    critic = _critic_and_detect(job_id, idxs, blocks, iteration, jlog, total)

    while iteration < config.DEEP_MAX_ITERATIONS:
        # Identify blocks that failed any of the three gates: critic, detector, similarity
        failing: dict[int, str] = {}
        for i in idxs:
            b = blocks[i]
            crit = b.get("critic_score") or 0
            det = b.get("detector_score") or 0
            sim = b.get("similarity_score")
            sim_failed = sim is not None and sim > config.DEEP_MAX_SIMILARITY_SCORE
            if crit < config.DEEP_TARGET_CRITIC_SCORE or det < config.DEEP_TARGET_DETECTOR_SCORE or sim_failed:
                hint = None
                for per in critic.get("per_chunk", []):
                    if per.get("idx") == i:
                        hint = per.get("suggested_fix")
                        break
                base = hint or "Rewrite in a more natural human voice. Remove AI cadence."
                if sim_failed:
                    base = (
                        "The previous attempt was too close to the original — reuse few of its word choices and break up its sentence boundaries. "
                        + base
                    )
                failing[i] = base
        if not failing:
            break
        iteration += 1
        needed_iteration = True
        jlog.info(f"iter={iteration} retrying {len(failing)} blocks")
        # Snapshot detector scores to check for improvement
        prev_dets = {i: blocks[i].get("detector_score") or 0 for i in failing}
        _rephrase_block_set(job_id, list(failing.keys()), blocks, voice_profile, user_samples, iteration, failing, total)
        critic = _critic_and_detect(job_id, list(failing.keys()), blocks, iteration, jlog, total)
        # Anti-regression: stop iterating on blocks that didn't improve
        any_improved = any(
            (blocks[i].get("detector_score") or 0) > prev_dets[i] for i in failing
        )
        if not any_improved:
            jlog.info(f"iter={iteration} no improvement on any failing block, accepting best")
            break

    for i in idxs:
        _pick_best(i, blocks)
        _emit_block(job_id, blocks[i], total)
    return needed_iteration


def run(job_id: str):
    """Deep-mode entry point. Called by rephraser dispatcher in a background thread."""
    t0 = time.time()
    doc = store.get(job_id)
    if not doc:
        log.error(f"[job:{job_id[:8]}] job not found")
        return

    run_path = rd.make_run_dir()
    doc["run_dir"] = str(run_path)
    store.update(job_id, run_dir=str(run_path))

    jlog = setup_job_logger(job_id, run_path, store)
    jlog.info(f"run dir: {run_path}  mode: deep")

    try:
        store.update(job_id, status="running")
        blocks: list[dict] = doc["blocks"]
        prose_total = sum(1 for b in blocks if b["type"] == "prose")
        jlog.info(
            f"start: {len(blocks)} blocks ({prose_total} prose, "
            f"{sum(1 for b in blocks if b['type'] == 'header')} headers, "
            f"{sum(1 for b in blocks if b['type'] == 'list')} lists, "
            f"{sum(1 for b in blocks if b['type'] == 'code')} code, "
            f"{sum(1 for b in blocks if b['type'] == 'math')} math)"
        )
        rd.save_text(run_path, "input.txt", doc.get("raw_text", structure.reassemble(blocks)))

        # 1. Voice calibration
        user_samples: Optional[list[str]] = doc.get("user_samples")
        if user_samples:
            jlog.info(f"voice calibration from {len(user_samples)} user sample(s)...")
            doc["voice_profile"] = llm_client.extract_voice_from_samples(user_samples, job_id=job_id)
        else:
            jlog.info("voice calibration from document paragraphs...")
            prose_texts = [b["original"] for b in blocks if b["type"] == "prose"]
            sample = random.sample(prose_texts, min(config.VOICE_SAMPLE_SIZE, len(prose_texts))) if prose_texts else []
            # Voice profile from text samples (reuses sentence-style extractor — paragraphs work fine as input)
            doc["voice_profile"] = llm_client.extract_voice_profile(sample, job_id=job_id) if sample else {"field_guess": "unknown", "tone_notes": "n/a"}
        rd.save_json(run_path, "voice_profile.json", doc["voice_profile"])
        store.emit(job_id, "progress", {"stage": "voice_profile", "voice_profile": doc["voice_profile"]})
        jlog.info(
            f"voice: field={doc['voice_profile'].get('field_guess')} "
            f"tone={doc['voice_profile'].get('tone_notes')}"
        )

        # 2. Triage prose blocks via blind detector
        prose_idxs = structure.prose_block_indexes(blocks)
        if prose_idxs:
            jlog.info(f"triage: scoring {len(prose_idxs)} prose blocks with blind detector...")
            store.emit(job_id, "progress", {"stage": "triage", "total": len(prose_idxs)})
            _triage_prose_blocks(job_id, prose_idxs, blocks, prose_total, jlog)

        # 3. Intent extraction only for flagged blocks
        flagged = [i for i in prose_idxs if blocks[i]["needs_rewrite"]]
        passthrough_count = len(prose_idxs) - len(flagged)
        jlog.info(f"triage result: {passthrough_count} passthrough, {len(flagged)} to rewrite")
        if flagged:
            jlog.info("extracting paragraph-level intents...")
            store.emit(job_id, "progress", {"stage": "intent", "total": len(flagged)})
            _extract_intents(job_id, flagged, blocks, jlog)

        # 4. Rephrase flagged blocks in batches
        any_needed_iteration = False
        processed_since_checkpoint = 0
        for start in range(0, len(flagged), config.DEEP_CHUNK_BATCH_SIZE):
            batch = flagged[start : start + config.DEEP_CHUNK_BATCH_SIZE]
            jlog.info(f"batch: blocks {batch}")
            if _process_chunk_batch(job_id, batch, blocks, doc["voice_profile"], user_samples, jlog, prose_total):
                any_needed_iteration = True
            doc["progress"]["done"] = sum(1 for b in blocks if b["type"] == "prose" and b["status"] in ("accepted", "passthrough"))
            processed_since_checkpoint += len(batch)
            if processed_since_checkpoint >= config.CHECKPOINT_EVERY_BLOCKS:
                _checkpoint(run_path, doc, jlog)
                processed_since_checkpoint = 0

        _checkpoint(run_path, doc, jlog)

        # 5. Final coherence pass (optional)
        if any_needed_iteration or not config.DEEP_SKIP_FINAL_PASS_IF_ALL_FIRST_TRY:
            jlog.info("final coherence pass...")
            store.emit(job_id, "progress", {"stage": "final_coherence"})
            assembled = structure.reassemble(blocks)
            try:
                final_doc = llm_client.final_coherence_pass(assembled, doc["voice_profile"], job_id=job_id)
                doc["final_output"] = final_doc
            except Exception as e:
                jlog.warning(f"final pass failed, using assembled blocks as-is: {e}")
                doc["final_output"] = assembled
        else:
            jlog.info("skipping final coherence pass (nothing required iteration)")
            doc["final_output"] = structure.reassemble(blocks)

        rd.save_text(run_path, "output.txt", doc["final_output"])
        rd.save_json(run_path, "doc.json", doc)

        duration = time.time() - t0
        accepted = [b for b in blocks if b["status"] == "accepted"]
        passthrough = [b for b in blocks if b["status"] == "passthrough" and b["type"] == "prose"]
        avg_det = (
            sum((b.get("detector_score") or 0) for b in accepted) / max(1, len(accepted))
            if accepted else None
        )
        usage_summary = usage.tracker.summary(job_id)
        rd.save_json(run_path, "usage.json", usage_summary)
        store.update(job_id, status="done")
        store.emit(
            job_id,
            "done",
            {
                "duration_sec": round(duration, 1),
                "blocks_total": len(blocks),
                "prose_total": prose_total,
                "rewritten": len(accepted),
                "passthrough": len(passthrough),
                "avg_detector_score": round(avg_det, 2) if avg_det is not None else None,
                "run_dir": str(run_path),
                "usage": usage_summary,
            },
        )
        jlog.info(
            f"complete blocks={len(blocks)} prose={prose_total} "
            f"rewritten={len(accepted)} passthrough={len(passthrough)} "
            f"avg_detector={avg_det if avg_det is None else round(avg_det, 2)} "
            f"duration={duration:.1f}s"
        )
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
            _checkpoint(run_path, doc, jlog)
            jlog.info("checkpoint saved on error — partial output recoverable in checkpoint.txt")
        except Exception:
            pass
        store.update(job_id, status="error", error=str(e))
        store.emit(job_id, "error", {"message": str(e)})
