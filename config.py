"""Central config for the document rephraser.

Edit values here to tune the pipeline. Everything (models, context window,
batch size, iteration thresholds, parallelism) lives in this one file.
"""

# ---------------- Models ----------------
# Cross-family asymmetric setup:
#   Writers/analyzers = Anthropic Claude (consistent voice, same family)
#   Critic            = Google Gemini   (independent judge, different blind spots)
MODEL_INTENT = "claude-haiku-4-5"  # Claude — fast cheap intent extraction
MODEL_REPHRASER = "claude-sonnet-4-6"  # Claude, main rewriter
MODEL_VOICE_PROFILE = (
    "claude-opus-4-7"  # Claude — voice calibration (aligns with writer)
)
MODEL_FINAL_PASS = "claude-sonnet-4-6"  # Claude — whole-document coherence

MODEL_CRITIC = "gemini-3.1-pro-preview"  # Gemini — cross-family critic (prompted, structured)
CRITIC_PROVIDER = "gemini"  # "gemini" or "anthropic"

# Blind AI detector — separate UNPRIMED call, no rubric, deliberately different signal.
# Used in deep mode as the real acceptance gate (the critic Goodharts against its own rubric).
MODEL_DETECTOR = "gemini-3.1-pro-preview"
DETECTOR_PROVIDER = "gemini"

# ---------------- Mode ----------------
DEFAULT_MODE = "deep"  # "quick" (sentence-level) or "deep" (block-level + triage + detector)

# ---------------- Context window ----------------
CONTEXT_BEFORE = 3  # sentences before target fed as context
CONTEXT_AFTER = 3  # sentences after target fed as context

# ---------------- Critic loop ----------------
BATCH_SIZE = 5  # sentences per critic pass
MAX_ITERATIONS = 3  # max rewrite attempts per batch
TARGET_AI_SCORE = 8  # ai_likelihood_score >= this passes (0-10, 10 = human)
TARGET_FLOW_SCORE = 7  # batch_flow_score >= this passes (0-10)

# ---------------- Voice calibration ----------------
VOICE_SAMPLE_SIZE = 5  # random sentences sampled to infer voice profile

# ---------------- Parallelism ----------------
PARALLEL_WORKERS = 8  # concurrent intent + rephrase workers

# ---------------- Persistence / checkpoints ----------------
RUNS_DIR = "runs"        # root directory for all run artifacts
CHECKPOINT_EVERY = 20    # quick mode: save checkpoint.txt every N processed sentences
CHECKPOINT_EVERY_BLOCKS = 5  # deep mode: save checkpoint.txt every N processed blocks

# ---------------- Deep mode ----------------
# Chunk-level pipeline with structure preservation, detector-triage, and dual-signal gate.
DEEP_CHUNK_BATCH_SIZE = 3            # paragraphs per critic call (paragraphs are larger than sentences)
DEEP_PARAGRAPH_CONTEXT_BEFORE = 1    # paragraphs before target fed as context
DEEP_PARAGRAPH_CONTEXT_AFTER = 1
DEEP_PASSTHROUGH_SCORE = 7           # if blind detector >= this on the original, skip rewriting
DEEP_TARGET_DETECTOR_SCORE = 7       # rewrite must achieve this on blind detector to accept
DEEP_TARGET_CRITIC_SCORE = 7         # rewrite must achieve this on prompted critic to accept
DEEP_MAX_SIMILARITY_SCORE = 5        # rewrite must score AT MOST this on similarity (0=thoroughly rewritten, 10=near-copy)
DEEP_MAX_ITERATIONS = 3
DEEP_SKIP_FINAL_PASS_IF_ALL_FIRST_TRY = True  # skip the doc-level pass when nothing needed iteration

# ---------------- Intensity presets ----------------
# User-facing knob: how aggressively to rewrite. Higher intensity = rewrite more paragraphs,
# require more meaningful change, accept fewer near-copies, iterate more.
INTENSITY_PRESETS = {
    "light": {
        # Skip paragraphs aggressively; loose targets; one shot per paragraph.
        "DEEP_PASSTHROUGH_SCORE": 8,
        "DEEP_TARGET_DETECTOR_SCORE": 6,
        "DEEP_TARGET_CRITIC_SCORE": 6,
        "DEEP_MAX_SIMILARITY_SCORE": 7,
        "DEEP_MAX_ITERATIONS": 1,
        "TARGET_AI_SCORE": 7,
        "TARGET_FLOW_SCORE": 6,
        "MAX_ITERATIONS": 1,
    },
    "balanced": {
        # The system defaults (current behavior).
        "DEEP_PASSTHROUGH_SCORE": 7,
        "DEEP_TARGET_DETECTOR_SCORE": 7,
        "DEEP_TARGET_CRITIC_SCORE": 7,
        "DEEP_MAX_SIMILARITY_SCORE": 5,
        "DEEP_MAX_ITERATIONS": 3,
        "TARGET_AI_SCORE": 8,
        "TARGET_FLOW_SCORE": 7,
        "MAX_ITERATIONS": 3,
    },
    "aggressive": {
        # Rewrite almost everything; demand strong human-ness and big surface change.
        "DEEP_PASSTHROUGH_SCORE": 4,
        "DEEP_TARGET_DETECTOR_SCORE": 8,
        "DEEP_TARGET_CRITIC_SCORE": 8,
        "DEEP_MAX_SIMILARITY_SCORE": 3,
        "DEEP_MAX_ITERATIONS": 4,
        "TARGET_AI_SCORE": 9,
        "TARGET_FLOW_SCORE": 8,
        "MAX_ITERATIONS": 4,
    },
}
DEFAULT_INTENSITY = "balanced"

# ---------------- Token limits ----------------
MAX_TOKENS_INTENT = 400
MAX_TOKENS_REPHRASE = 500
MAX_TOKENS_VOICE_PROFILE = 800
MAX_TOKENS_CRITIC = 2000
MAX_TOKENS_FINAL_PASS = 8000
