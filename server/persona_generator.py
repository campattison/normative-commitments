"""
persona_generator.py — PDF text extraction and philosopher persona generation.

Uses PyMuPDF (fitz) for PDF text extraction and Claude (via claude_bridge)
for generating structured persona JSON from extracted text.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import fitz  # PyMuPDF

from belief_extractor import parse_json_from_llm
from claude_bridge import query_claude

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str, max_chars: int = 200_000) -> str:
    """Extract text from a PDF file, capped at *max_chars* characters.

    Uses PyMuPDF for fast, layout-aware extraction.  Returns the
    concatenated text of all pages (up to the character limit).
    """
    parts: list[str] = []
    total = 0

    with fitz.open(pdf_path) as doc:
        for page in doc:
            try:
                text = page.get_text()
            except Exception:
                logger.warning("Failed to extract page %d, skipping", page.number)
                continue
            if total + len(text) > max_chars:
                parts.append(text[: max_chars - total])
                break
            parts.append(text)
            total += len(text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Persona generation prompt
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are an expert in the history of philosophy. Given the following extracted text from a philosophical work, generate a structured persona profile for the philosopher who wrote it.

The persona will be used in a Socratic dialogue system where users can discuss their normative commitments with various philosophers. The persona must capture the philosopher's distinctive voice, method, and positions.

**Return ONLY valid JSON** with this exact schema:

```json
{{
  "name": "Full Name",
  "tradition": "Philosophical tradition (e.g., Kantian Constructivism, Utilitarianism, Virtue Ethics)",
  "summary": "One sentence summary of who this philosopher is and what they're known for.",
  "key_positions": ["Position 1", "Position 2", "Position 3", "Position 4", "Position 5"],
  "system_prompt": "The full system prompt (2000-4000 characters)...",
  "contest_prompt": "The full contest mode prompt (2000-4000 characters)..."
}}
```

**Guidelines for the system_prompt:**
- Write in second person: "You are [Name], the [title/affiliation]..."
- Capture their distinctive philosophical voice and method
- List 6-8 specific behavioral instructions as bullet points
- Reference their actual works, concepts, and terminology
- Include a "Tone:" line at the end
- Follow this structural pattern:

```
You are [Name], [title]. You are engaged in a serious philosophical dialogue with a fellow philosopher who is working through their normative commitments.

Your role:
- [Specific behavioral instruction referencing their actual philosophy]
- [Another instruction]
...

Tone: [2-3 adjectives]. [Brief elaboration].
```

**Guidelines for the contest_prompt:**
- Same philosopher voice, but focused on helping revise a belief
- Instruction to understand what the interlocutor wants to revise and why
- Probe whether the revision is philosophically motivated
- Reference connected beliefs and downstream consequences
- Be collaborative but rigorous
- Include a "Tone:" line at the end

**Important:**
- Base everything on the actual text provided — do not fabricate positions not supported by the source
- The philosopher's voice should be distinctive and recognizable
- Key positions should be specific claims, not vague generalities

Here is the extracted text:

---

{text}

---

Return ONLY the JSON object. No commentary before or after."""


def generate_persona(
    extracted_text: str,
    source_filename: str,
    reading_path: str,
) -> dict | None:
    """Generate a philosopher persona from extracted PDF text.

    Calls Claude to produce structured JSON, then enriches it with
    metadata fields (id, source info, timestamps).

    Returns the complete persona dict or None on failure.
    """
    prompt = GENERATION_PROMPT.format(text=extracted_text[:80_000])

    raw = query_claude(prompt)
    if raw.startswith("[ERROR]"):
        logger.error("Claude query failed during persona generation: %s", raw)
        return None

    parsed = parse_json_from_llm(raw)

    if parsed is None:
        logger.error("Failed to parse persona JSON from Claude response")
        return None

    # Validate required fields
    required = ("name", "system_prompt", "contest_prompt")
    for field in required:
        if field not in parsed or not parsed[field]:
            logger.error("Missing required field '%s' in generated persona", field)
            return None

    # Generate a slug ID from the philosopher's name
    name = parsed["name"]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", source_filename.lower().replace(".pdf", "")).strip("-")

    return {
        "id": slug,
        "name": name,
        "source_pdf": source_filename,
        "source_reading": reading_path,
        "created": datetime.now(timezone.utc).isoformat(),
        "system_prompt": parsed["system_prompt"],
        "contest_prompt": parsed["contest_prompt"],
        "summary": parsed.get("summary", ""),
        "key_positions": parsed.get("key_positions", []),
        "tradition": parsed.get("tradition", ""),
        "builtin": False,
    }


# ---------------------------------------------------------------------------
# Structured reading generation prompt
# ---------------------------------------------------------------------------

READING_PROMPT = """You are an expert philosophical reader creating structured reading notes for a graduate-level philosophical workbench. Given the extracted text from a philosophical work, produce a detailed, analytically structured reading document in Markdown.

**About the philosopher:**
- Name: {name}
- Tradition: {tradition}
- Key positions: {key_positions}

**About the reader:**
{user_positions}

**Instructions:**

Produce a reading document with these sections:

## Overview
One paragraph situating the work: what it is, when/where it was published or delivered, what its central project is, and why it matters in the philosophical landscape.

## Section-by-Section Summaries
Follow the actual structure of the text (chapters, lectures, sections — whatever the work uses). For each:
- A descriptive heading matching the original
- A substantive summary of the argument (not just topic — capture the moves)
- Note key distinctions, definitions, and argumentative pivots

## Key Quotes
5-10 verbatim passages that capture the work's most important claims and distinctive voice. For each:
- The exact quote in a blockquote
- A parenthetical reference to where it appears (section, page, or paragraph if visible)
- A brief note on why this passage matters

## Key Tensions
How this philosopher's positions interact with the reader's existing commitments. Consider:
- Where do they challenge the reader's existing views?
- What genuine tensions emerge that will need to be worked through?

Be specific — reference particular arguments from the text and particular commitments from the reader's profile. Don't be generic.

---

Here is the extracted text:

{text}

---

Produce the full reading document now. Write in an analytical, precise voice. Be substantive — these notes should help the reader engage deeply with the philosopher's arguments without re-reading the full text."""


def generate_structured_reading(
    extracted_text: str,
    persona: dict,
    user_positions: str,
) -> Optional[str]:
    """Generate a structured reading document from extracted PDF text.

    Uses the generated persona metadata for context and the user's starting
    positions to map tensions. Returns the reading as Markdown.
    """
    key_positions_str = "\n".join(
        f"  - {p}" for p in persona.get("key_positions", [])
    )

    prompt = READING_PROMPT.format(
        name=persona.get("name", "Unknown"),
        tradition=persona.get("tradition", ""),
        key_positions=key_positions_str,
        user_positions=user_positions,
        text=extracted_text[:80_000],
    )

    result = query_claude(prompt)

    if result.startswith("[ERROR]"):
        logger.error("Structured reading generation failed: %s", result)
        return None

    return result
