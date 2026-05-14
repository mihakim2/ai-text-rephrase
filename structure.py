"""Document structure parsing and reassembly.

Splits a .txt document into typed blocks so we only rephrase prose paragraphs
and pass everything else (headers, lists, code, math, blank lines) through verbatim.

Block types:
  prose   — running text we will rephrase
  header  — markdown headings or short bare-line titles
  list    — bullet or numbered lists (passthrough)
  code    — triple-backtick fenced code blocks (passthrough)
  math    — display math fenced by $$...$$ (passthrough)
  blank   — paragraph separator (preserved as one blank line)
"""
from __future__ import annotations

import re
from typing import Literal, TypedDict

BlockType = Literal["prose", "header", "list", "code", "math", "blank"]

_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_HEADER_HASH = re.compile(r"^\s{0,3}#{1,6}\s+\S")


class Block(TypedDict, total=False):
    id: int
    type: BlockType
    original: str         # canonical source text for this block (verbatim slice for non-prose)
    final_text: str       # what ends up in the output
    needs_rewrite: bool   # only meaningful for prose; set by detector triage
    detector_original: dict | None  # blind detector score on the original
    intent: dict | None
    rephrased_versions: list[dict]
    critic_score: int | None
    detector_score: int | None
    iterations_used: int
    status: str           # pending | passthrough | rephrased | accepted | error


def parse_blocks(text: str) -> list[Block]:
    """Walk lines and emit typed blocks. Idempotent: reassemble(parse_blocks(t)) == t modulo trailing whitespace."""
    if not text:
        return []
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    n = len(lines)
    next_id = 0

    def push(btype: BlockType, content: str):
        nonlocal next_id
        blocks.append({
            "id": next_id,
            "type": btype,
            "original": content,
            "final_text": content,           # default: passthrough
            "needs_rewrite": False,
            "detector_original": None,
            "intent": None,
            "rephrased_versions": [],
            "critic_score": None,
            "detector_score": None,
            "iterations_used": 0,
            "status": "passthrough" if btype != "prose" else "pending",
        })
        next_id += 1

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block (```...```)
        if stripped.startswith("```"):
            start = i
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            # Include the closing fence if present
            end = j if j < n else j - 1
            push("code", "\n".join(lines[start : end + 1]))
            i = end + 1
            continue

        # Display math: single-line ($$ ... $$) or multi-line ($$ ... \n ... $$)
        if stripped.startswith("$$"):
            if len(stripped) > 2 and stripped.endswith("$$"):
                # Single-line display math
                push("math", line)
                i += 1
                continue
            start = i
            j = i + 1
            while j < n and lines[j].strip() != "$$":
                j += 1
            end = j if j < n else j - 1
            push("math", "\n".join(lines[start : end + 1]))
            i = end + 1
            continue

        # Blank line(s) — collapse consecutive blanks to one paragraph break
        if not stripped:
            push("blank", "")
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            continue

        # Markdown ATX header
        if _HEADER_HASH.match(line):
            push("header", line)
            i += 1
            continue

        # Collect a paragraph (until next blank line or end)
        start = i
        j = i
        while j < n and lines[j].strip():
            j += 1
        paragraph = "\n".join(lines[start:j])
        # Classify
        para_lines = [pl for pl in paragraph.split("\n") if pl.strip()]
        if para_lines and sum(bool(_LIST_LINE.match(pl)) for pl in para_lines) / len(para_lines) > 0.5:
            push("list", paragraph)
        elif _looks_like_header(paragraph):
            push("header", paragraph)
        else:
            push("prose", paragraph)
        i = j

    return blocks


def _looks_like_header(paragraph: str) -> bool:
    """Heuristic: a single short line without terminal punctuation = header.

    Avoids false positives on one-sentence paragraphs that just don't end with a period.
    Tightened to require <= 80 chars AND <= 12 words AND no terminal punctuation.
    """
    if "\n" in paragraph:
        return False
    p = paragraph.strip()
    if len(p) > 80:
        return False
    if len(p.split()) > 12:
        return False
    if p.endswith((".", "!", "?", ":", ";", ",")):
        return False
    return True


def reassemble(blocks: list[Block]) -> str:
    """Join blocks back into a single document, preserving paragraph separators."""
    out: list[str] = []
    prev_type: BlockType | None = None
    for b in blocks:
        if b["type"] == "blank":
            out.append("")
            prev_type = "blank"
            continue
        # Insert a paragraph break between two non-blank blocks if the source didn't
        if prev_type is not None and prev_type != "blank" and out and out[-1] != "":
            out.append("")
        out.append(b.get("final_text") or b["original"])
        prev_type = b["type"]
    return "\n".join(out).rstrip() + "\n"


def split_sentences_in_block(text: str) -> list[str]:
    """Light wrapper so the rephraser can show sentence-level previews if needed."""
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in seg.segment(text) if s and s.strip()]


def prose_block_indexes(blocks: list[Block]) -> list[int]:
    return [i for i, b in enumerate(blocks) if b["type"] == "prose"]
