"""
belief_extractor.py — Automatic belief extraction from dialogue.

A passive reader that analyzes philosophical dialogue and extracts
the human speaker's beliefs, updating the graph without altering
the dialogue experience.

Supports a rich diff schema: new nodes/edges, confidence updates,
label updates, edge updates, edge removals, retractions, and
tension identification.  Designed for batch processing of multiple
accumulated turns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from claude_bridge import query_claude

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All valid confidence levels (including "retracted")
# ---------------------------------------------------------------------------

VALID_CONFIDENCES = frozenset({
    "certain", "probable", "tentative", "under-revision", "retracted",
})

# ---------------------------------------------------------------------------
# Content gate
# ---------------------------------------------------------------------------


def should_extract(history: list[dict]) -> bool:
    """Return False only for exchanges with zero philosophical content.

    The real filtering happens in the LLM — this gate exists only to
    skip purely procedural messages (greetings, "ok", empty turns).
    """
    # Need at least one user message
    user_msgs = [m for m in history if m.get("role") in ("user", "human")]
    if not user_msgs:
        return False

    last_user = user_msgs[-1].get("content", "").strip()

    # Skip only truly empty / procedural messages (< 10 chars with no
    # philosophical content markers)
    if len(last_user) < 10:
        philosophical_markers = (
            "think", "believe", "agree", "disagree", "matter", "should",
            "ought", "wrong", "right", "true", "false", "value", "moral",
            "duty", "pain", "reason", "norm", "commit", "reject", "accept",
        )
        if not any(marker in last_user.lower() for marker in philosophical_markers):
            return False

    return True


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a philosophical belief extraction system. Your task is to analyze NEW dialogue turns between a human and a philosopher, and extract the HUMAN speaker's philosophical beliefs and commitments.

## Current Belief Graph

{graph_state}

## Dialogue to Analyze

{dialogue}

## Instructions

1. Extract ONLY the human speaker's beliefs — never the philosopher's statements or questions.
2. Only extract philosophical, ethical, or metaethical positions. Ignore logistical remarks, software discussion, or small talk.
3. **Check existing nodes by MEANING, not wording.** If the human restates an existing belief ("I think outcomes matter" ≈ existing "consequentialism" node), update its confidence — do NOT create a duplicate.
4. Prefer updating confidence on existing nodes over creating new ones.
5. **If the human retracts a position** ("Actually, I no longer think X"), add it to `retracted_nodes`.
6. **Contradictions are allowed and valuable.** If the human holds two positions that are in tension, add both and include a `tensions_identified` entry. These are sites for future philosophical work.
7. **Err toward extraction.** If the human's position is ambiguous, extract it as `tentative`. If nothing at all is extractable from these turns, include a `skip_reason` field explaining why.
8. For new nodes, generate a slug-style ID (lowercase, hyphens, e.g. "value-of-autonomy").
9. If the human says "I agree" or confirms a position the philosopher articulated, that counts as a belief — update or add accordingly.

## Confidence Mapping
- "certain": firmly stated and defended under pressure
- "probable": stated with confidence but not yet tested
- "tentative": exploratory, hedged, or speculative
- "under-revision": actively questioning a previously held position
- "retracted": explicitly abandoned (use only via retracted_nodes)

## Edge Weight Scale
- 0.3: weak or implied connection
- 0.5: moderate, stated but not elaborated
- 0.7: strong, explicitly articulated
- 0.9: logical entailment or definitional

## Response Format

Return ONLY a JSON object with this exact structure (no markdown fences, no explanation):

{{"new_nodes": [{{"id": "slug-id", "label": "Human-Readable Label", "type": "belief", "tier": "metaethics|normative|applied", "confidence": "certain|probable|tentative|under-revision"}}], "new_edges": [{{"source": "existing-or-new-node-id", "target": "existing-or-new-node-id", "polarity": "support|tension", "weight": 0.3, "description": "Why this connection exists"}}], "confidence_updates": [{{"node_id": "existing-node-id", "new_confidence": "under-revision", "reason": "Why confidence changed"}}], "label_updates": [{{"node_id": "existing-node-id", "new_label": "Revised Human-Readable Label", "reason": "Why label was refined"}}], "edge_updates": [{{"edge_id": "existing-edge-id", "new_weight": 0.3, "reason": "Why weight changed"}}], "edge_removals": [{{"edge_id": "existing-edge-id", "reason": "Why this connection no longer holds"}}], "retracted_nodes": [{{"node_id": "existing-node-id", "reason": "Why the human retracted this", "successor_id": "optional-new-node-id-or-null"}}], "tensions_identified": [{{"between": ["node-a-id", "node-b-id"], "description": "Why these two positions are in tension"}}]}}

If nothing is extractable, return: {{"new_nodes": [], "new_edges": [], "confidence_updates": [], "label_updates": [], "edge_updates": [], "edge_removals": [], "retracted_nodes": [], "tensions_identified": [], "skip_reason": "Brief explanation"}}
"""


def _format_graph_state(graph: dict) -> str:
    """Format the current graph for the extraction prompt."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return "(Graph is currently empty — no existing beliefs.)"

    parts = ["### Nodes"]
    for n in nodes:
        nid = n.get("id", "?")
        label = n.get("label", "?")
        conf = n.get("confidence", "?")
        tier = n.get("tier", "?")
        parts.append(f"- {nid}: \"{label}\" (tier={tier}, confidence={conf})")

    if edges:
        parts.append("\n### Edges")
        for e in edges:
            eid = e.get("id", "?")
            src = e.get("source", "?")
            tgt = e.get("target", "?")
            pol = e.get("polarity", "?")
            wt = e.get("weight", "?")
            desc = e.get("description", "")
            parts.append(f"- {eid}: {src} → {tgt} ({pol}, weight={wt}): {desc}")

    return "\n".join(parts)


def _format_dialogue(history: list[dict]) -> str:
    """Format dialogue history for the extraction prompt (full history)."""
    lines = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "human"):
            lines.append(f"Human: {content}")
        else:
            lines.append(f"Philosopher: {content}")
    return "\n\n".join(lines)


def _format_turn_window(history: list[dict], turn_indices: list[int]) -> str:
    """Format ONLY the turns in the batch window, with turn numbers.

    Falls back to the full history if turn_indices is empty or invalid.
    """
    if not turn_indices:
        return _format_dialogue(history)

    # Clamp indices to valid range
    valid_indices = [i for i in turn_indices if 0 <= i < len(history)]
    if not valid_indices:
        return _format_dialogue(history)

    lines = ["## New dialogue turns (since last extraction)\n"]
    for idx in sorted(valid_indices):
        m = history[idx]
        role = m.get("role", "user")
        content = m.get("content", "")
        speaker = "Human" if role in ("user", "human") else "Philosopher"
        lines.append(f"[Turn {idx}] {speaker}: {content}")

    return "\n\n".join(lines)


def build_extraction_prompt(
    graph: dict,
    history: list[dict],
    turn_indices: list[int] | None = None,
) -> str:
    """Build the full extraction prompt with current graph and dialogue.

    If turn_indices is provided, only those turns are formatted as the
    dialogue window (batch mode).  Otherwise the full history is used.
    """
    graph_state = _format_graph_state(graph)
    if turn_indices:
        dialogue = _format_turn_window(history, turn_indices)
    else:
        dialogue = _format_dialogue(history)

    return EXTRACTION_PROMPT.format(
        graph_state=graph_state,
        dialogue=dialogue,
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_json_from_llm(raw: str) -> dict | None:
    """Parse a JSON object from raw LLM output, tolerating markdown fences.

    Uses string-aware brace matching to extract the outermost JSON object.
    Returns the parsed dict or None on failure.
    """
    if not raw or raw.startswith("[ERROR]"):
        logger.warning("LLM query failed: %s", raw[:200] if raw else "(empty)")
        return None

    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Find a JSON object via brace matching
    start = text.find("{")
    if start == -1:
        logger.warning("No JSON object found in LLM response.")
        return None

    # Find the matching closing brace (accounting for strings)
    depth = 0
    end = start
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    try:
        result = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None

    return result


# All expected keys in an extraction response
_EXTRACTION_LIST_KEYS = (
    "new_nodes", "new_edges", "confidence_updates",
    "label_updates", "edge_updates", "edge_removals",
    "retracted_nodes", "tensions_identified",
)


def parse_extraction_response(raw: str) -> dict | None:
    """Parse an extraction-specific JSON response from the LLM.

    Returns the parsed dict with validated extraction keys, or None on failure.
    """
    result = parse_json_from_llm(raw)
    if result is None:
        return None

    # Ensure all list keys exist
    for key in _EXTRACTION_LIST_KEYS:
        if key not in result or not isinstance(result[key], list):
            result[key] = []

    return result


# ---------------------------------------------------------------------------
# Diff validation
# ---------------------------------------------------------------------------


def validate_diff(diff: dict, existing_graph: dict) -> dict:
    """Validate and clean the extraction diff.

    Handles all eight operation types.  Returns a cleaned diff with
    invalid items removed.
    """
    existing_ids = {n["id"] for n in existing_graph.get("nodes", []) if "id" in n}
    existing_edge_ids = {e["id"] for e in existing_graph.get("edges", []) if "id" in e}
    new_ids: set[str] = set()
    _valid_id_re = re.compile(r"^[a-z0-9][a-z0-9-]*$")

    # --- new_nodes ---
    valid_nodes: list[dict] = []
    for node in diff.get("new_nodes", []):
        nid = node.get("id")
        if not nid or not isinstance(nid, str) or not _valid_id_re.match(nid):
            logger.info("Skipping node with invalid ID: %r", nid)
            continue
        if nid in existing_ids:
            logger.info("Skipping duplicate node: %s", nid)
            continue
        node.setdefault("label", nid.replace("-", " ").title())
        node.setdefault("type", "belief")
        node.setdefault("tier", "normative")
        node.setdefault("confidence", "tentative")
        # Validate confidence — new nodes cannot be "retracted"
        conf = node.get("confidence")
        if conf not in VALID_CONFIDENCES or conf == "retracted":
            node["confidence"] = "tentative"
        valid_nodes.append(node)
        new_ids.add(nid)

    all_ids = existing_ids | new_ids

    # --- new_edges ---
    valid_edges: list[dict] = []
    for edge in diff.get("new_edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt:
            continue
        if src not in all_ids or tgt not in all_ids:
            logger.info("Skipping edge with unknown node: %s → %s", src, tgt)
            continue
        polarity = edge.get("polarity", "support")
        edge_id = f"e-{src}-{tgt}-{polarity}"
        if edge_id in existing_edge_ids:
            logger.info("Skipping duplicate edge: %s", edge_id)
            continue
        edge["id"] = edge_id
        edge.setdefault("polarity", "support")
        edge.setdefault("description", "")
        # Validate edge weight is a number in [0, 1]
        w = edge.get("weight")
        if not isinstance(w, (int, float)) or not (0.0 <= w <= 1.0):
            edge["weight"] = 0.5
        valid_edges.append(edge)
        existing_edge_ids.add(edge_id)  # prevent duplicates within batch

    # --- confidence_updates ---
    valid_confidence: list[dict] = []
    for upd in diff.get("confidence_updates", []):
        nid = upd.get("node_id")
        if not nid or nid not in all_ids:
            logger.info("Skipping confidence update for unknown node: %s", nid)
            continue
        if upd.get("new_confidence") not in VALID_CONFIDENCES:
            continue
        valid_confidence.append(upd)

    # --- label_updates ---
    valid_labels: list[dict] = []
    for upd in diff.get("label_updates", []):
        nid = upd.get("node_id")
        new_label = upd.get("new_label")
        if not nid or nid not in all_ids:
            logger.info("Skipping label update for unknown node: %s", nid)
            continue
        if not new_label or not isinstance(new_label, str) or not new_label.strip():
            continue
        valid_labels.append(upd)

    # --- edge_updates ---
    valid_edge_updates: list[dict] = []
    for upd in diff.get("edge_updates", []):
        eid = upd.get("edge_id")
        if not eid or eid not in existing_edge_ids:
            logger.info("Skipping edge update for unknown edge: %s", eid)
            continue
        new_weight = upd.get("new_weight")
        if new_weight is not None:
            if not isinstance(new_weight, (int, float)) or not (0.0 <= new_weight <= 1.0):
                logger.info("Skipping edge update with invalid weight: %s = %r", eid, new_weight)
                continue
        valid_edge_updates.append(upd)

    # --- edge_removals ---
    valid_edge_removals: list[dict] = []
    for rem in diff.get("edge_removals", []):
        eid = rem.get("edge_id")
        if not eid or eid not in existing_edge_ids:
            logger.info("Skipping edge removal for unknown edge: %s", eid)
            continue
        valid_edge_removals.append(rem)

    # --- retracted_nodes ---
    valid_retractions: list[dict] = []
    for ret in diff.get("retracted_nodes", []):
        nid = ret.get("node_id")
        if not nid or nid not in existing_ids:
            logger.info("Skipping retraction for unknown node: %s", nid)
            continue
        successor = ret.get("successor_id")
        if successor and successor not in all_ids:
            logger.info("Retraction successor %s not found — clearing", successor)
            ret["successor_id"] = None
        valid_retractions.append(ret)

    # --- tensions_identified ---
    valid_tensions: list[dict] = []
    for tens in diff.get("tensions_identified", []):
        between = tens.get("between")
        if not isinstance(between, list) or len(between) != 2:
            continue
        if between[0] not in all_ids or between[1] not in all_ids:
            logger.info("Skipping tension with unknown nodes: %s", between)
            continue
        valid_tensions.append(tens)

    return {
        "new_nodes": valid_nodes,
        "new_edges": valid_edges,
        "confidence_updates": valid_confidence,
        "label_updates": valid_labels,
        "edge_updates": valid_edge_updates,
        "edge_removals": valid_edge_removals,
        "retracted_nodes": valid_retractions,
        "tensions_identified": valid_tensions,
    }


# ---------------------------------------------------------------------------
# Diff application
# ---------------------------------------------------------------------------


def _diff_is_empty(diff: dict) -> bool:
    """Return True if no operations are present in the diff."""
    return all(len(diff.get(k, [])) == 0 for k in _EXTRACTION_LIST_KEYS)


def apply_diff(diff: dict, graph_transaction) -> dict:
    """Apply a validated diff to graph.json using the graph transaction manager.

    Handles all eight operation types.
    Returns the applied diff (for reporting to the frontend).
    """
    if _diff_is_empty(diff):
        return diff

    with graph_transaction() as graph:
        nodes_by_id = {n["id"]: n for n in graph["nodes"]}
        edges_by_id = {e["id"]: e for e in graph["edges"]}

        # 1. Append new nodes
        for node in diff.get("new_nodes", []):
            graph["nodes"].append(node)
            nodes_by_id[node["id"]] = node
            logger.info("Added node: %s (%s)", node["id"], node["label"])

        # 2. Append new edges
        for edge in diff.get("new_edges", []):
            graph["edges"].append(edge)
            edges_by_id[edge["id"]] = edge
            logger.info("Added edge: %s", edge["id"])

        # 3. Confidence updates
        for upd in diff.get("confidence_updates", []):
            node = nodes_by_id.get(upd["node_id"])
            if node:
                old = node.get("confidence", "?")
                node["confidence"] = upd["new_confidence"]
                logger.info(
                    "Updated confidence: %s %s → %s (%s)",
                    upd["node_id"], old, upd["new_confidence"],
                    upd.get("reason", ""),
                )

        # 4. Label updates
        for upd in diff.get("label_updates", []):
            node = nodes_by_id.get(upd["node_id"])
            if node:
                old = node.get("label", "?")
                node["label"] = upd["new_label"]
                logger.info(
                    "Updated label: %s \"%s\" → \"%s\" (%s)",
                    upd["node_id"], old, upd["new_label"],
                    upd.get("reason", ""),
                )

        # 5. Edge weight updates
        for upd in diff.get("edge_updates", []):
            edge = edges_by_id.get(upd["edge_id"])
            if edge:
                old_w = edge.get("weight", "?")
                if "new_weight" in upd:
                    edge["weight"] = upd["new_weight"]
                logger.info(
                    "Updated edge: %s weight %s → %s (%s)",
                    upd["edge_id"], old_w, upd.get("new_weight", old_w),
                    upd.get("reason", ""),
                )

        # 6. Edge removals
        removed_edge_ids = {r["edge_id"] for r in diff.get("edge_removals", [])}
        if removed_edge_ids:
            graph["edges"] = [
                e for e in graph["edges"] if e.get("id") not in removed_edge_ids
            ]
            for eid in removed_edge_ids:
                logger.info("Removed edge: %s", eid)

        # 7. Retracted nodes — set confidence to "retracted", optionally
        #    transfer incoming edges to successor
        for ret in diff.get("retracted_nodes", []):
            node = nodes_by_id.get(ret["node_id"])
            if node:
                old = node.get("confidence", "?")
                node["confidence"] = "retracted"
                logger.info(
                    "Retracted node: %s (was %s) — %s",
                    ret["node_id"], old, ret.get("reason", ""),
                )
                successor_id = ret.get("successor_id")
                if successor_id and successor_id in nodes_by_id:
                    # Transfer incoming edges to the successor
                    for edge in graph["edges"]:
                        if edge.get("target") == ret["node_id"]:
                            edge["target"] = successor_id
                            logger.info(
                                "Transferred edge %s target → %s",
                                edge.get("id", "?"), successor_id,
                            )

        # 8. Tensions — add as tension edges between the two nodes
        for tens in diff.get("tensions_identified", []):
            between = tens["between"]
            src, tgt = between[0], between[1]
            edge_id = f"e-{src}-{tgt}-tension"
            # Skip if this tension edge already exists
            if edge_id not in edges_by_id:
                tension_edge = {
                    "id": edge_id,
                    "source": src,
                    "target": tgt,
                    "polarity": "tension",
                    "weight": 0.5,
                    "description": tens.get("description", "Identified tension"),
                }
                graph["edges"].append(tension_edge)
                edges_by_id[edge_id] = tension_edge
                logger.info(
                    "Added tension edge: %s ↔ %s — %s",
                    src, tgt, tens.get("description", ""),
                )

    return diff


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


def run_extraction(
    history: list[dict],
    read_graph,
    graph_transaction,
    turn_indices: list[int] | None = None,
) -> dict | None:
    """Run the full extraction pipeline.

    Args:
        history: Dialogue history [{role, content}, ...]
        read_graph: Callable that returns current graph dict
        graph_transaction: Context manager for atomic graph writes
        turn_indices: If provided, only these turns are shown to the LLM
            as the extraction window (batch mode).

    Returns:
        The applied diff dict, or None if extraction was skipped/failed.
    """
    current_graph = read_graph()
    prompt = build_extraction_prompt(current_graph, history, turn_indices)

    n_turns = len(turn_indices) if turn_indices else len(history)
    logger.info("Running belief extraction (%d turns in window)...", n_turns)
    raw_response = query_claude(prompt)

    parsed = parse_extraction_response(raw_response)
    if parsed is None:
        logger.warning("Extraction parse failed — skipping.")
        return None

    validated = validate_diff(parsed, current_graph)

    if _diff_is_empty(validated):
        skip_reason = parsed.get("skip_reason", "")
        logger.info(
            "Extraction found nothing new.%s",
            f" Reason: {skip_reason}" if skip_reason else "",
        )
        return validated

    applied = apply_diff(validated, graph_transaction)
    counts = {k: len(applied.get(k, [])) for k in _EXTRACTION_LIST_KEYS}
    logger.info("Extraction applied: %s", counts)
    return applied


# ---------------------------------------------------------------------------
# Candidate extraction (lightweight, non-committing)
# ---------------------------------------------------------------------------

CANDIDATE_PROMPT = """\
You are a philosophical belief extraction system operating in CANDIDATE mode. \
Your task is to suggest possible graph updates — these will be reviewed before \
being applied.

## Current Belief Graph

{graph_state}

## Full Dialogue Context

{dialogue_context}

## Recent Exchanges (focus window)

{recent_turns}

## Instructions

Analyze the RECENT EXCHANGES above (using the full dialogue for context) and \
suggest AT MOST 5 candidate updates to the belief graph.

**Extract ONLY the HUMAN speaker's beliefs** — never the philosopher's \
statements, questions, or hypotheticals.

Each candidate must be one of three types:

1. **new_node** — A belief not yet represented in the graph.
   Required fields: `type` ("new_node"), `id` (slug-style, lowercase with \
hyphens), `label` (human-readable), `tier` ("metaethics", "normative", or \
"applied"), `confidence` ("certain", "probable", "tentative", or \
"under-revision"), `rationale` (why you believe the human holds this), \
`suggested_edges` (list of connections to existing nodes, each with `source`, \
`target`, `polarity` ("support" or "tension"), and `weight` (0.3–0.9)).

2. **confidence_update** — A change in how strongly the human holds an \
existing belief.
   Required fields: `type` ("confidence_update"), `node_id` (existing node), \
`new_confidence` ("certain", "probable", "tentative", "under-revision", or \
"retracted"), `rationale`.

3. **tension** — A newly apparent tension between two existing beliefs.
   Required fields: `type` ("tension"), `between` (list of exactly 2 existing \
node IDs), `description` (what the tension is), `rationale` (evidence from \
dialogue).

## Response Format

Return ONLY a JSON object with this exact structure (no markdown fences, \
no explanation):

{{"candidates": [{{"type": "new_node|confidence_update|tension", ...}}]}}

If nothing is extractable, return: {{"candidates": []}}
"""


def extract_candidates(
    history: list[dict],
    read_graph,
    turn_window: int = 3,
) -> list[dict] | None:
    """Extract candidate graph updates without committing them.

    Analyzes the most recent exchanges against the full dialogue context
    and current graph state, returning up to 5 validated candidates for
    human review.

    Args:
        history: Dialogue history [{role, content}, ...]
        read_graph: Callable that returns current graph dict
        turn_window: Number of exchanges (user+assistant pairs) to use
            as the focus window.  Defaults to 3.

    Returns:
        A list of validated candidate dicts, or None if the query failed.
    """
    graph = read_graph()
    graph_state = _format_graph_state(graph)
    dialogue_context = _format_dialogue(history)

    # Build the recent-turns window: last turn_window*2 messages
    n_recent = turn_window * 2
    recent_messages = history[-n_recent:] if len(history) > n_recent else history
    recent_lines = []
    for m in recent_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        speaker = "Human" if role in ("user", "human") else "Philosopher"
        recent_lines.append(f"{speaker}: {content}")
    recent_turns = "\n\n".join(recent_lines)

    prompt = CANDIDATE_PROMPT.format(
        graph_state=graph_state,
        dialogue_context=dialogue_context,
        recent_turns=recent_turns,
    )

    logger.info(
        "Running candidate extraction (window=%d exchanges, %d recent msgs)...",
        turn_window, len(recent_messages),
    )
    raw = query_claude(prompt)

    parsed = parse_json_from_llm(raw)
    if parsed is None:
        logger.warning("Candidate extraction parse failed.")
        return None

    raw_candidates = parsed.get("candidates")
    if not isinstance(raw_candidates, list):
        logger.warning("Candidate response missing 'candidates' list.")
        return None

    # Build lookup sets for validation
    existing_ids = {n["id"] for n in graph.get("nodes", []) if "id" in n}
    _valid_id_re = re.compile(r"^[a-z0-9][a-z0-9-]*$")

    valid: list[dict] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        ctype = candidate.get("type")

        if ctype == "new_node":
            cid = candidate.get("id")
            if not cid or not isinstance(cid, str) or not _valid_id_re.match(cid):
                logger.info("Skipping candidate new_node with invalid ID: %r", cid)
                continue
            if cid in existing_ids:
                logger.info("Skipping candidate new_node — ID already exists: %s", cid)
                continue
            if not candidate.get("label"):
                logger.info("Skipping candidate new_node without label: %s", cid)
                continue
            valid.append(candidate)

        elif ctype == "confidence_update":
            nid = candidate.get("node_id")
            if not nid or nid not in existing_ids:
                logger.info(
                    "Skipping confidence_update for unknown node: %s", nid,
                )
                continue
            if candidate.get("new_confidence") not in VALID_CONFIDENCES:
                logger.info(
                    "Skipping confidence_update with invalid confidence: %r",
                    candidate.get("new_confidence"),
                )
                continue
            valid.append(candidate)

        elif ctype == "tension":
            between = candidate.get("between")
            if not isinstance(between, list) or len(between) != 2:
                logger.info("Skipping tension with invalid 'between' field.")
                continue
            if between[0] not in existing_ids or between[1] not in existing_ids:
                logger.info(
                    "Skipping tension with unknown nodes: %s", between,
                )
                continue
            valid.append(candidate)

        else:
            logger.info("Skipping candidate with unknown type: %r", ctype)

    # Hard cap
    valid = valid[:5]

    logger.info("Candidate extraction produced %d valid candidates.", len(valid))
    return valid
