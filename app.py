import json
import logging
import threading

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

import config as cfg
import structure
from job_store import store
from logging_config import setup_logging, job_log
from rephraser import run_job, split_sentences

load_dotenv()
setup_logging()
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    pasted = (request.form.get("text") or "").strip()
    source_filename: str
    raw: str

    if pasted:
        if len(pasted) < 20:
            return jsonify({"error": "pasted text is too short"}), 400
        raw = pasted
        source_filename = "pasted.txt"
    elif "file" in request.files:
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400
        if not f.filename.lower().endswith(".txt"):
            return jsonify({"error": "only .txt files supported"}), 400
        try:
            raw = f.read().decode("utf-8", errors="replace")
        except Exception as e:
            return jsonify({"error": f"could not read file: {e}"}), 400
        source_filename = f.filename
    else:
        return jsonify({"error": "provide either a file upload or pasted text"}), 400

    mode = (request.form.get("mode") or cfg.DEFAULT_MODE).lower()
    if mode not in ("quick", "deep"):
        return jsonify({"error": "mode must be 'quick' or 'deep'"}), 400

    samples_raw = (request.form.get("samples") or "").strip()
    user_samples = None
    if samples_raw and mode == "deep":
        # Allow either '\n---\n' separators or just paragraph breaks
        if "---" in samples_raw:
            user_samples = [s.strip() for s in samples_raw.split("---") if s.strip()]
        else:
            user_samples = [p.strip() for p in samples_raw.split("\n\n") if p.strip()]

    intensity = (request.form.get("intensity") or cfg.DEFAULT_INTENSITY).lower()
    if intensity not in cfg.INTENSITY_PRESETS:
        intensity = cfg.DEFAULT_INTENSITY
    thresholds = dict(cfg.INTENSITY_PRESETS[intensity])

    cfg_snapshot = {
        "mode": mode,
        "intensity": intensity,
        "thresholds": thresholds,
    }

    if mode == "quick":
        sentences = split_sentences(raw)
        if not sentences:
            return jsonify({"error": "no sentences found in document"}), 400
        job_id = store.create_quick(source_filename, sentences, cfg_snapshot)
        log.info(job_log(job_id, f"uploaded mode=quick source='{source_filename}' sentences={len(sentences)}"))
        return jsonify({
            "job_id": job_id,
            "mode": "quick",
            "unit_count": len(sentences),
            "unit_label": "sentences",
        }), 201

    # deep mode
    blocks = structure.parse_blocks(raw)
    prose_count = sum(1 for b in blocks if b["type"] == "prose")
    if not blocks:
        return jsonify({"error": "empty document"}), 400
    if prose_count == 0:
        return jsonify({"error": "no prose blocks found — nothing to rephrase"}), 400
    job_id = store.create_deep(source_filename, raw, blocks, user_samples, cfg_snapshot)
    log.info(job_log(
        job_id,
        f"uploaded mode=deep source='{source_filename}' blocks={len(blocks)} prose={prose_count} "
        f"samples={len(user_samples) if user_samples else 0}"
    ))
    return jsonify({
        "job_id": job_id,
        "mode": "deep",
        "unit_count": len(blocks),
        "prose_count": prose_count,
        "unit_label": "blocks",
        "block_types": [b["type"] for b in blocks],
    }), 201


@app.route("/api/jobs/<job_id>/start", methods=["POST"])
def start(job_id):
    doc = store.get(job_id)
    if not doc:
        return jsonify({"error": "job not found"}), 404
    if doc["status"] not in ("pending",):
        return jsonify({"error": f"job already {doc['status']}"}), 400
    t = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"status": "running"}), 202


@app.route("/api/jobs/<job_id>/stream")
def stream(job_id):
    doc = store.get(job_id)
    if not doc:
        return jsonify({"error": "job not found"}), 404

    @stream_with_context
    def gen():
        for msg in store.stream(job_id, timeout=15.0):
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/jobs/<job_id>")
def job_info(job_id):
    doc = store.get(job_id)
    if not doc:
        return jsonify({"error": "job not found"}), 404
    return jsonify(
        {
            "job_id": job_id,
            "status": doc["status"],
            "progress": doc["progress"],
            "source_filename": doc["source_filename"],
            "error": doc.get("error"),
        }
    )


@app.route("/api/jobs/<job_id>/changes")
def changes(job_id):
    doc = store.get(job_id)
    if not doc:
        return jsonify({"error": "job not found"}), 404
    if doc.get("mode") == "deep":
        items = []
        for b in doc["blocks"]:
            if b["type"] == "blank":
                continue
            items.append({
                "id": b["id"],
                "type": b["type"],
                "status": b["status"],
                "original": b["original"],
                "final": b.get("final_text") or b["original"],
                "detector_score": b.get("detector_score"),
                "critic_score": b.get("critic_score"),
                "similarity_score": b.get("similarity_score"),
                "iterations": b.get("iterations_used", 0),
            })
        return jsonify({"mode": "deep", "items": items})
    items = [{
        "id": s["id"],
        "status": s["status"],
        "original": s["original"],
        "final": s.get("final_text") or s["original"],
        "critic_score": s.get("current_score"),
        "iterations": s.get("iterations_used", 0),
    } for s in doc.get("sentences", [])]
    return jsonify({"mode": "quick", "items": items})


@app.route("/api/jobs/<job_id>/download")
def download(job_id):
    doc = store.get(job_id)
    if not doc:
        return jsonify({"error": "job not found"}), 404
    if doc["status"] != "done":
        return jsonify({"error": f"job not done (status={doc['status']})"}), 400
    text = doc.get("final_output") or store.final_text(job_id)
    base = doc["source_filename"].rsplit(".", 1)[0]
    return Response(
        text,
        mimetype="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{base}_humanized.txt"'
        },
    )


@app.route("/api/jobs/<job_id>/debug")
def debug(job_id):
    return Response(store.to_json(job_id), mimetype="application/json")


def _local_addresses(port: int) -> list[str]:
    """Best-effort: return reachable URLs (loopback, LAN, Tailscale)."""
    import socket
    urls = [f"http://127.0.0.1:{port}"]
    seen = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip in seen:
                continue
            seen.add(ip)
            label = "tailscale" if ip.startswith("100.") else "lan"
            urls.append(f"http://{ip}:{port}  ({label})")
    except Exception:
        pass
    # Probe a UDP socket to the default route — reveals the primary LAN IP even
    # if getaddrinfo missed it.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip not in seen:
            urls.append(f"http://{ip}:{port}  (lan)")
            seen.add(ip)
    except Exception:
        pass
    return urls


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5055"))
    host = os.environ.get("HOST", "0.0.0.0")  # listen on all interfaces by default
    log.info(f"binding {host}:{port}  (override with HOST= / PORT=)")
    for url in _local_addresses(port):
        log.info(f"  reachable at  {url}")
    log.warning("server is accessible to anyone who can reach this machine on the network — "
                "API key usage will be billed to your account. Set HOST=127.0.0.1 to restrict to local only.")
    app.run(host=host, port=port, debug=False, threaded=True)
