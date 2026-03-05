"""
app.py — Flask application for the Normative Commitments workbench.

Serves the single-page UI, exposes a file API for reading and writing
markdown documents, and provides a streaming dialogue endpoint that
bridges to Claude CLI via claude_bridge.py.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime

from flask import Flask, Response, jsonify, request, send_from_directory

from claude_bridge import get_status, query_claude, stream_dialogue

from belief_extractor import extract_candidates, parse_json_from_llm, run_extraction, should_extract

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.realpath(os.path.join(SERVER_DIR, ".."))
UI_DIR = os.path.join(PROJECT_ROOT, "ui")

# Content directories that the file API is allowed to scan.
CONTENT_DIRS = ["commitments", "tensions", "dialogues", "readings"]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------


def _validate_path(filepath: str) -> tuple[str, str | None]:
    """Resolve *filepath* relative to PROJECT_ROOT and verify it stays inside.

    Returns (resolved_absolute_path, error_message | None).
    If error_message is not None the request should be rejected.
    """
    # Join with project root, then resolve all symlinks / ..
    resolved = os.path.realpath(os.path.join(PROJECT_ROOT, filepath))

    if not resolved.startswith(PROJECT_ROOT + os.sep) and resolved != PROJECT_ROOT:
        return resolved, "Path traversal denied."

    # Block access to the server directory itself (no reading app.py, etc.)
    server_resolved = os.path.realpath(SERVER_DIR)
    if resolved.startswith(server_resolved + os.sep) or resolved == server_resolved:
        return resolved, "Access to server internals is denied."

    return resolved, None


# ---------------------------------------------------------------------------
# Persona / system prompt loading
# ---------------------------------------------------------------------------


def _load_system_prompt() -> str:
    """Load the project-level CLAUDE.md as the base persona prompt.

    Falls back to a sensible default if the file does not exist yet.
    """
    claude_md_path = os.path.join(PROJECT_ROOT, "CLAUDE.md")
    if os.path.isfile(claude_md_path):
        with open(claude_md_path, "r", encoding="utf-8") as f:
            return f.read()

    # Graceful fallback — the project CLAUDE.md hasn't been created yet.
    return (
        "You are a rigorous philosopher engaging in Socratic dialogue. "
        "Draw on the Kantian constructivist tradition (especially Korsgaard) "
        "to probe, clarify, and stress-test the user's normative commitments. "
        "Be precise, charitable, and incisive."
    )


def _extract_user_positions() -> str:
    """Extract the user's starting positions from CLAUDE.md.

    Looks for a '## Starting Positions' or '## *'s Starting Positions'
    section. Returns the section text, or a generic fallback if the file
    or section is not found.
    """
    claude_md_path = os.path.join(PROJECT_ROOT, "CLAUDE.md")
    if not os.path.isfile(claude_md_path):
        return _USER_POSITIONS_FALLBACK

    with open(claude_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match "## Starting Positions" or "## Someone's Starting Positions"
    match = re.search(r"(## (?:\w+'s )?Starting Positions.*?)(?=\n#{1,3} |\Z)", content, re.DOTALL)
    if not match:
        return _USER_POSITIONS_FALLBACK

    return match.group(1).strip()


_USER_POSITIONS_FALLBACK = """\
No starting positions configured. Add a '## Starting Positions' section \
to CLAUDE.md in the project root to personalize reading notes and dialogue."""


# ---------------------------------------------------------------------------
# Routes — Static serving
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main UI entry point."""
    return send_from_directory(UI_DIR, "index.html")


@app.route("/ui/<path:filename>")
def serve_ui(filename: str):
    """Serve static assets (CSS, JS, images) from the ui/ directory."""
    return send_from_directory(UI_DIR, filename)


# ---------------------------------------------------------------------------
# Routes — File API
# ---------------------------------------------------------------------------


@app.route("/api/files")
def list_files():
    """List all .md files across the content directories.

    Returns a JSON array of {path, name, category} objects, where *path*
    is relative to PROJECT_ROOT.
    """
    results: list[dict] = []

    for category in CONTENT_DIRS:
        cat_dir = os.path.join(PROJECT_ROOT, category)
        if not os.path.isdir(cat_dir):
            continue
        for root, _dirs, files in os.walk(cat_dir):
            for fname in sorted(files):
                if not fname.endswith(".md"):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
                results.append(
                    {
                        "path": rel_path,
                        "name": fname,
                        "category": category,
                    }
                )

    return jsonify(results)


@app.route("/api/files/<path:filepath>", methods=["GET"])
def read_file(filepath: str):
    """Read a single markdown file and return its content."""
    resolved, error = _validate_path(filepath)
    if error:
        return jsonify({"error": error}), 403

    if not os.path.isfile(resolved):
        return jsonify({"error": "File not found."}), 404

    with open(resolved, "r", encoding="utf-8") as f:
        content = f.read()

    return jsonify({"path": filepath, "content": content})


@app.route("/api/files/<path:filepath>", methods=["POST"])
def write_file(filepath: str):
    """Write or update a markdown file.

    Expects JSON body: {"content": "..."}
    Creates parent directories if they do not exist.
    """
    resolved, error = _validate_path(filepath)
    if error:
        return jsonify({"error": error}), 403

    data = request.get_json(silent=True)
    if not data or "content" not in data:
        return jsonify({"error": "Request body must include 'content'."}), 400

    # Ensure parent directory exists.
    parent = os.path.dirname(resolved)
    os.makedirs(parent, exist_ok=True)

    with open(resolved, "w", encoding="utf-8") as f:
        f.write(data["content"])

    return jsonify({"path": filepath, "status": "saved"})


# ---------------------------------------------------------------------------
# Routes — Dialogue
# ---------------------------------------------------------------------------


@app.route("/api/dialogue", methods=["POST"])
def dialogue():
    """Streaming dialogue endpoint (Server-Sent Events).

    Expects JSON body:
        {
            "system_prompt": "optional override — merged with CLAUDE.md",
            "history": [{"role": "human"|"assistant", "content": "..."}],
            "message": "the new user turn"
        }

    Streams SSE events:
        data: <chunk>\n\n
        ...
        data: [DONE]\n\n
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Request body must include 'message'."}), 400

    # Build system prompt: project persona + optional per-request context.
    base_prompt = _load_system_prompt()
    extra_context = data.get("system_prompt", "")
    if extra_context:
        system_prompt = f"{base_prompt}\n\n---\n\n{extra_context}"
    else:
        system_prompt = base_prompt

    history = data.get("history", [])
    user_message = data["message"]

    def generate():
        for chunk in stream_dialogue(system_prompt, history, user_message):
            # Each SSE event is "data: ...\n\n".
            # We send each chunk (typically a line) as its own event.
            escaped = chunk.rstrip("\n")
            yield f"data: {escaped}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/dialogue/save", methods=["POST"])
def save_dialogue():
    """Persist a dialogue transcript to the dialogues/ directory.

    Expects JSON body:
        {
            "topic": "Human-readable topic string",
            "messages": [
                {"role": "human"|"assistant", "content": "..."},
                ...
            ]
        }

    Writes to dialogues/YYYY-MM-DD_topic-slug.md
    """
    data = request.get_json(silent=True)
    if not data or "topic" not in data or "messages" not in data:
        return jsonify({"error": "Request body must include 'topic' and 'messages'."}), 400

    topic: str = data["topic"]
    messages: list[dict] = data["messages"]

    # Build a filesystem-safe slug from the topic.
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60]
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{slug}.md"
    rel_path = os.path.join("dialogues", filename)

    resolved, error = _validate_path(rel_path)
    if error:
        return jsonify({"error": error}), 403

    # Format the transcript as readable markdown.
    lines: list[str] = [
        f"# Dialogue: {topic}",
        f"*Recorded {date_str}*",
        "",
    ]
    for msg in messages:
        role = msg.get("role", "human")
        content = msg.get("content", "")
        if role in ("human", "user"):
            lines.append(f"## Human\n\n{content}\n")
        else:
            lines.append(f"## Philosopher\n\n{content}\n")

    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return jsonify({"path": rel_path, "status": "saved"})


# ---------------------------------------------------------------------------
# Graph I/O (thread-safe)
# ---------------------------------------------------------------------------

GRAPH_PATH = os.path.join(PROJECT_ROOT, "graph.json")
_graph_lock = threading.Lock()

_EMPTY_GRAPH: dict = {"version": 1, "nodes": [], "edges": []}


@contextlib.contextmanager
def _graph_transaction():
    """Hold the graph lock for a full read-modify-write cycle."""
    with _graph_lock:
        if not os.path.isfile(GRAPH_PATH):
            graph = dict(_EMPTY_GRAPH, nodes=[], edges=[])
        else:
            try:
                with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                    graph = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("graph.json is corrupt — resetting to empty graph.")
                graph = dict(_EMPTY_GRAPH, nodes=[], edges=[])
        yield graph
        # Atomic write via temp file + rename
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(GRAPH_PATH), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, GRAPH_PATH)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def _read_graph() -> dict:
    """Read graph.json (snapshot), returning a default empty graph if missing."""
    with _graph_lock:
        if not os.path.isfile(GRAPH_PATH):
            return dict(_EMPTY_GRAPH, nodes=[], edges=[])
        try:
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("graph.json is corrupt — returning empty graph.")
            return dict(_EMPTY_GRAPH, nodes=[], edges=[])


# ---------------------------------------------------------------------------
# Routes — Graph API
# ---------------------------------------------------------------------------


@app.route("/api/graph", methods=["GET"])
def get_graph():
    """Return the full graph JSON."""
    return jsonify(_read_graph())


@app.route("/api/graph", methods=["POST"])
def post_graph():
    """Atomic overwrite of the entire graph."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        return jsonify({"error": "Graph must contain 'nodes' and 'edges' arrays."}), 400
    with _graph_transaction() as graph:
        graph.clear()
        graph.update(data)
    return jsonify({"status": "saved"})


@app.route("/api/graph/node", methods=["PATCH"])
def patch_node():
    """Add or update a single node.  Expects JSON with at least 'id'."""
    data = request.get_json(silent=True)
    if not data or "id" not in data:
        return jsonify({"error": "Request body must include 'id'."}), 400

    with _graph_transaction() as graph:
        existing = next((n for n in graph["nodes"] if n["id"] == data["id"]), None)
        if existing:
            existing.update(data)
        else:
            graph["nodes"].append(data)

    return jsonify({"status": "saved", "node": data["id"]})


@app.route("/api/graph/node/<node_id>", methods=["DELETE"])
def delete_node(node_id: str):
    """Remove a node and all edges connected to it."""
    # Check existence first (read-only), then mutate in a transaction.
    snapshot = _read_graph()
    if not any(n["id"] == node_id for n in snapshot["nodes"]):
        return jsonify({"error": f"Node '{node_id}' not found."}), 404

    with _graph_transaction() as graph:
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != node_id]
        graph["edges"] = [
            e for e in graph["edges"]
            if e["source"] != node_id and e["target"] != node_id
        ]
    return jsonify({"status": "deleted", "node": node_id})


@app.route("/api/graph/edge", methods=["PATCH"])
def patch_edge():
    """Add or update a single edge.  Expects JSON with at least 'id'."""
    data = request.get_json(silent=True)
    if not data or "id" not in data:
        return jsonify({"error": "Request body must include 'id'."}), 400

    with _graph_transaction() as graph:
        existing = next((e for e in graph["edges"] if e["id"] == data["id"]), None)
        if existing:
            existing.update(data)
        else:
            graph["edges"].append(data)

    return jsonify({"status": "saved", "edge": data["id"]})


@app.route("/api/graph/edge/<edge_id>", methods=["DELETE"])
def delete_edge(edge_id: str):
    """Remove a single edge by ID."""
    snapshot = _read_graph()
    if not any(e["id"] == edge_id for e in snapshot["edges"]):
        return jsonify({"error": f"Edge '{edge_id}' not found."}), 404

    with _graph_transaction() as graph:
        graph["edges"] = [e for e in graph["edges"] if e["id"] != edge_id]
    return jsonify({"status": "deleted", "edge": edge_id})


# ---------------------------------------------------------------------------
# Routes — Belief Extraction (queue + drain pattern)
# ---------------------------------------------------------------------------

# Queue of unprocessed turn indices — appended by the endpoint, drained by
# the extraction thread.  Protected by _pending_lock.
_pending_turns: list[int] = []
_pending_history: list[dict] = []       # latest full history for context
_pending_lock = threading.Lock()        # protects queue reads/writes

# NOTE: _extraction_running is acquired by extract_beliefs() and released
# by _drain_extraction_queue()'s finally block. This acquire-in-caller /
# release-in-callee pattern ensures exactly-one-at-a-time execution.
_extraction_running = threading.Lock()

# NOTE: These globals work correctly only in single-process deployments
# (Flask dev server, or gunicorn with --threads but 1 worker).
_latest_extraction: dict | None = None
_latest_extraction_ts: float = 0.0
_extraction_result_lock = threading.Lock()


def _drain_extraction_queue() -> None:
    """Drain all queued turns, batching them into one extraction call.

    Loops until the queue is empty — if new turns arrive while extraction
    runs, they are picked up in the next iteration.
    """
    global _latest_extraction, _latest_extraction_ts

    try:
        while True:
            # Grab everything currently queued
            with _pending_lock:
                if not _pending_turns:
                    return  # nothing left — done
                turn_indices = list(_pending_turns)
                history = list(_pending_history)
                _pending_turns.clear()

            if not should_extract(history):
                logger.info("Queued turns skipped — trivial content.")
                continue

            logger.info(
                "Draining extraction queue: %d turns (indices %s)",
                len(turn_indices), turn_indices,
            )

            try:
                result = run_extraction(
                    history, _read_graph, _graph_transaction,
                    turn_indices=turn_indices,
                )
                with _extraction_result_lock:
                    _latest_extraction = result
                    _latest_extraction_ts = datetime.now().timestamp()
            except Exception:
                logger.exception("Belief extraction failed for batch %s", turn_indices)
    finally:
        _extraction_running.release()


@app.route("/api/extract-beliefs", methods=["POST"])
def extract_beliefs():
    """Queue a turn for belief extraction.

    Accepts {history, new_turn_index}.  The turn is always queued
    (never silently dropped).  Returns 202 immediately; the frontend
    polls /api/extract-beliefs/latest for results.
    """
    data = request.get_json(silent=True)
    if not data or "history" not in data:
        return jsonify({"error": "Request body must include 'history'."}), 400

    history = data["history"]
    new_turn_index = data.get("new_turn_index")

    # Default: use last two turns (the user message + assistant response)
    if new_turn_index is None:
        new_turn_index = max(0, len(history) - 2)

    # Always enqueue — never drop
    with _pending_lock:
        # Enqueue each turn index from the new exchange onward
        for idx in range(new_turn_index, len(history)):
            if idx not in _pending_turns:
                _pending_turns.append(idx)
        # Always store the latest full history for context
        _pending_history.clear()
        _pending_history.extend(history)

    logger.info(
        "Enqueued turn(s) starting at %d (queue depth: %d)",
        new_turn_index, len(_pending_turns),
    )

    # Try to start the drain loop
    acquired = _extraction_running.acquire(blocking=False)
    if not acquired:
        # Extraction is already running — turns are safely queued and will
        # be picked up when the current extraction finishes its drain loop.
        return jsonify({"status": "queued", "reason": "extraction in progress"}), 202

    try:
        t = threading.Thread(
            target=_drain_extraction_queue,
            daemon=True,
        )
        t.start()
    except Exception:
        _extraction_running.release()
        logger.exception("Failed to start extraction thread")
        return jsonify({"error": "Internal error"}), 500

    return jsonify({"status": "accepted"}), 202


@app.route("/api/extract-beliefs/latest", methods=["GET"])
def get_latest_extraction():
    """Return the most recent extraction result."""
    with _extraction_result_lock:
        return jsonify({
            "result": _latest_extraction,
            "timestamp": _latest_extraction_ts,
        })


@app.route("/api/extract-candidates", methods=["POST"])
def post_extract_candidates():
    """Extract candidate beliefs/insights from recent dialogue turns.

    Synchronous — the user clicked a button and is waiting.
    Returns {candidates: [...]} where each candidate has a type field.
    """
    body = request.get_json(silent=True) or {}
    history = body.get("history")
    if not history or not isinstance(history, list):
        return jsonify({"error": "history is required"}), 400

    turn_window = body.get("turn_window", 3)
    if not isinstance(turn_window, int) or turn_window < 1:
        turn_window = 3

    # Quick content gate — skip trivially empty exchanges
    if not should_extract(history):
        return jsonify({"candidates": [], "note": "No philosophical content detected"})

    candidates = extract_candidates(history, _read_graph, turn_window=turn_window)
    if candidates is None:
        return jsonify({"error": "Extraction failed"}), 502

    return jsonify({"candidates": candidates})


# ---------------------------------------------------------------------------
# Routes — Contest Node
# ---------------------------------------------------------------------------


def _build_contest_prompt(
    contested_node: dict,
    subgraph: dict,
    dialogue_history: list[dict],
) -> str:
    """Build the prompt for evaluating a belief revision's graph impact."""
    # Format the contested node
    node_desc = (
        f'Contested belief: "{contested_node.get("label", "?")}" '
        f'(id={contested_node.get("id", "?")}, '
        f'tier={contested_node.get("tier", "?")}, '
        f'confidence={contested_node.get("confidence", "?")})'
    )

    # Format subgraph context
    first_order_nodes = subgraph.get("first_order_nodes", [])
    second_order_nodes = subgraph.get("second_order_nodes", [])
    first_order_edges = subgraph.get("first_order_edges", [])
    second_order_edges = subgraph.get("second_order_edges", [])

    parts = [node_desc, ""]

    if first_order_nodes:
        parts.append("Directly connected beliefs:")
        for n in first_order_nodes:
            parts.append(
                f'  - "{n.get("label", "?")}" '
                f'(id={n.get("id", "?")}, confidence={n.get("confidence", "?")})'
            )

    if first_order_edges:
        parts.append("\nDirect connections:")
        for e in first_order_edges:
            parts.append(
                f"  - {e.get('source', '?')} → {e.get('target', '?')} "
                f"({e.get('polarity', '?')}, weight={e.get('weight', '?')})"
                f"{': ' + e['description'] if e.get('description') else ''}"
            )

    if second_order_nodes:
        parts.append("\nIndirectly connected beliefs (2nd order):")
        for n in second_order_nodes:
            parts.append(
                f'  - "{n.get("label", "?")}" '
                f'(id={n.get("id", "?")}, confidence={n.get("confidence", "?")})'
            )

    if second_order_edges:
        parts.append("\nIndirect connections:")
        for e in second_order_edges:
            parts.append(
                f"  - {e.get('source', '?')} → {e.get('target', '?')} "
                f"({e.get('polarity', '?')}, weight={e.get('weight', '?')})"
            )

    subgraph_text = "\n".join(parts)

    # Format dialogue
    dialogue_lines = []
    for m in dialogue_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "human"):
            dialogue_lines.append(f"Human: {content}")
        else:
            dialogue_lines.append(f"Philosopher: {content}")
    dialogue_text = "\n\n".join(dialogue_lines)

    return f"""\
You are a philosophical belief-revision evaluator. A philosopher has been contesting one of their beliefs through Socratic dialogue. Based on the dialogue and the belief graph context, evaluate what changes to the belief graph are warranted.

## Belief Graph Context

{subgraph_text}

## Dialogue About This Revision

{dialogue_text}

## Instructions

Evaluate the dialogue and determine what changes to the belief graph are warranted. Consider:
1. Should the contested belief's confidence change?
2. Should any directly connected beliefs have their confidence adjusted?
3. Should any edge weights change to reflect revised relationships?
4. Should any edges be removed (relationship no longer holds)?
5. Should any new beliefs or connections be added?

Be conservative — only propose changes that are clearly supported by the dialogue. If the human hasn't actually committed to a revision, propose minimal or no changes.

## Response Format

Return ONLY a JSON object with this exact structure (no markdown fences, no explanation):

{{"node_updates": [{{"id": "node-id", "confidence": "new-confidence", "label": "optional-new-label-or-null"}}], "edge_updates": [{{"id": "edge-id", "weight": 0.3, "description": "why changed"}}], "edge_deletions": ["edge-id-to-remove"], "new_nodes": [{{"id": "slug-id", "label": "Human-Readable Label", "type": "belief", "tier": "metaethics|normative|applied", "confidence": "tentative|probable|certain|under-revision"}}], "new_edges": [{{"source": "node-id", "target": "node-id", "polarity": "support|tension", "weight": 0.5, "description": "Why this connection"}}], "summary": "One paragraph explaining the overall revision."}}

If no changes are warranted, return: {{"node_updates": [], "edge_updates": [], "edge_deletions": [], "new_nodes": [], "new_edges": [], "summary": "No revision warranted — the dialogue did not produce a clear commitment to change."}}
"""


@app.route("/api/contest-node", methods=["POST"])
def contest_node():
    """Evaluate a belief revision's impact on the graph.

    Synchronous — gathers 1st/2nd order subgraph, sends to Claude, returns
    a structured revision diff.

    Expects JSON body:
        {
            "node_id": "contested-node-id",
            "history": [{"role": "user"|"philosopher", "content": "..."}]
        }
    """
    data = request.get_json(silent=True)
    if not data or "node_id" not in data or "history" not in data:
        return jsonify({"error": "Request body must include 'node_id' and 'history'."}), 400

    node_id = data["node_id"]
    history = data["history"]

    graph = _read_graph()
    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    contested = nodes_by_id.get(node_id)
    if not contested:
        return jsonify({"error": f"Node '{node_id}' not found."}), 404

    # Gather 1st-order subgraph
    first_order_edges = []
    first_order_node_ids = set()
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if src == node_id or tgt == node_id:
            first_order_edges.append(e)
            first_order_node_ids.add(src)
            first_order_node_ids.add(tgt)
    first_order_node_ids.discard(node_id)

    # Gather 2nd-order subgraph
    second_order_edges = []
    second_order_node_ids = set()
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if e in first_order_edges:
            continue
        if src in first_order_node_ids or tgt in first_order_node_ids:
            second_order_edges.append(e)
            second_order_node_ids.add(src)
            second_order_node_ids.add(tgt)
    second_order_node_ids -= first_order_node_ids
    second_order_node_ids.discard(node_id)

    subgraph = {
        "first_order_nodes": [nodes_by_id[nid] for nid in first_order_node_ids if nid in nodes_by_id],
        "second_order_nodes": [nodes_by_id[nid] for nid in second_order_node_ids if nid in nodes_by_id],
        "first_order_edges": first_order_edges,
        "second_order_edges": second_order_edges,
    }

    prompt = _build_contest_prompt(contested, subgraph, history)

    try:
        raw_response = query_claude(prompt)
    except Exception:
        logger.exception("Contest evaluation failed")
        return jsonify({"error": "Failed to evaluate revision."}), 500

    parsed = parse_json_from_llm(raw_response)
    if parsed is None:
        return jsonify({"error": "Could not parse evaluation response."}), 500

    # Ensure all expected keys exist
    for key in ("node_updates", "edge_updates", "edge_deletions", "new_nodes", "new_edges"):
        if key not in parsed or not isinstance(parsed[key], list):
            parsed[key] = []
    if "summary" not in parsed or not isinstance(parsed["summary"], str):
        parsed["summary"] = ""

    return jsonify(parsed)


# ---------------------------------------------------------------------------
# Routes — Status
# ---------------------------------------------------------------------------


@app.route("/api/status")
def status():
    """Return Claude bridge status (active processes, capacity)."""
    return jsonify(get_status())


# ---------------------------------------------------------------------------
# Routes — Personas
# ---------------------------------------------------------------------------

PERSONAS_DIR = os.path.join(PROJECT_ROOT, "personas")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")


def _load_persona(persona_id: str) -> dict | None:
    """Load a persona JSON file by ID. Returns None if not found or corrupt."""
    safe_id = re.sub(r"[^a-z0-9_-]", "", persona_id.lower())
    if safe_id != persona_id:
        return None
    path = os.path.join(PERSONAS_DIR, f"{safe_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load persona %s: %s", persona_id, exc)
        return None


@app.route("/api/personas")
def list_personas():
    """List all personas (metadata only, no prompts).

    Returns sorted: builtin first, then alphabetical by name.
    """
    personas = []
    if not os.path.isdir(PERSONAS_DIR):
        return jsonify(personas)

    for fname in os.listdir(PERSONAS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(PERSONAS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping corrupt persona file %s: %s", fname, exc)
            continue
        # Return metadata only — omit system_prompt and contest_prompt
        personas.append({
            "id": data.get("id", fname[:-5]),
            "name": data.get("name", "Unknown"),
            "summary": data.get("summary", ""),
            "tradition": data.get("tradition", ""),
            "builtin": data.get("builtin", False),
        })

    # Sort: builtin first, then alphabetical by name
    personas.sort(key=lambda p: (not p["builtin"], p["name"].lower()))
    return jsonify(personas)


@app.route("/api/personas/<persona_id>")
def get_persona(persona_id: str):
    """Return full persona JSON including prompts."""
    # Sanitize ID to prevent path traversal
    safe_id = re.sub(r"[^a-z0-9_-]", "", persona_id.lower())
    if safe_id != persona_id:
        return jsonify({"error": "Invalid persona ID."}), 400

    data = _load_persona(persona_id)
    if data is None:
        return jsonify({"error": "Persona not found."}), 404
    return jsonify(data)


@app.route("/api/personas/<persona_id>", methods=["DELETE"])
def delete_persona(persona_id: str):
    """Delete a non-builtin persona."""
    safe_id = re.sub(r"[^a-z0-9_-]", "", persona_id.lower())
    if safe_id != persona_id:
        return jsonify({"error": "Invalid persona ID."}), 400

    data = _load_persona(persona_id)
    if data is None:
        return jsonify({"error": "Persona not found."}), 404
    if data.get("builtin", False):
        return jsonify({"error": "Cannot delete builtin personas."}), 403

    path = os.path.join(PERSONAS_DIR, f"{persona_id}.json")
    try:
        os.remove(path)
    except FileNotFoundError:
        return jsonify({"error": "Persona not found."}), 404
    except OSError as exc:
        logger.error("Failed to delete persona %s: %s", persona_id, exc)
        return jsonify({"error": "Failed to delete persona."}), 500
    return jsonify({"status": "deleted", "id": persona_id})


@app.route("/api/personas/generate", methods=["POST"])
def generate_persona_endpoint():
    """Upload a PDF and generate a philosopher persona from it.

    Expects multipart form with a 'pdf' file field.
    Returns 201 with the generated persona + reading path.
    """
    from persona_generator import extract_pdf_text, generate_persona, generate_structured_reading

    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF."}), 400

    # Save uploaded PDF
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    safe_filename = re.sub(r"[^a-zA-Z0-9._-]", "_", pdf_file.filename).lstrip(".")
    if not safe_filename:
        return jsonify({"error": "Invalid filename."}), 400
    upload_path = os.path.join(UPLOADS_DIR, safe_filename)
    pdf_file.save(upload_path)

    # Extract text
    try:
        text = extract_pdf_text(upload_path)
    except Exception:
        logger.exception("PDF extraction failed for %s", safe_filename)
        return jsonify({"error": "Failed to extract text from PDF."}), 500

    if not text or len(text.strip()) < 100:
        return jsonify({"error": "PDF appears to contain too little extractable text."}), 400

    # Generate slug for the reading
    base_slug = re.sub(r"[^a-z0-9]+", "_", safe_filename.lower().replace(".pdf", "")).strip("_")

    # Prepare reading path
    os.makedirs(os.path.join(PROJECT_ROOT, "readings"), exist_ok=True)
    reading_path = f"readings/{base_slug}.md"
    reading_abs = os.path.join(PROJECT_ROOT, reading_path)

    # Handle duplicate reading filenames
    counter = 1
    while os.path.exists(reading_abs):
        reading_path = f"readings/{base_slug}_{counter}.md"
        reading_abs = os.path.join(PROJECT_ROOT, reading_path)
        counter += 1

    # Generate persona
    try:
        persona = generate_persona(text, safe_filename, reading_path)
    except Exception:
        logger.exception("Persona generation failed for %s", safe_filename)
        return jsonify({"error": "Failed to generate persona from text."}), 500

    if persona is None:
        return jsonify({"error": "Could not generate a valid persona from this text."}), 500

    # Generate structured reading
    user_positions = _extract_user_positions()
    try:
        structured_reading = generate_structured_reading(text, persona, user_positions)
    except Exception:
        logger.exception("Structured reading generation failed for %s", safe_filename)
        structured_reading = None

    # Save reading — structured if available, raw fallback otherwise
    reading_quality = "structured"
    with open(reading_abs, "w", encoding="utf-8") as f:
        if structured_reading:
            f.write(structured_reading)
        else:
            reading_quality = "raw_fallback"
            logger.warning("Structured reading failed; saving raw text fallback for %s", safe_filename)
            preview = text[:10_000]
            suffix = "\n\n---\n\n*[Text truncated — structured reading generation failed]*" if len(text) > 10_000 else ""
            f.write(f"# {safe_filename}\n\n*Extracted from uploaded PDF (raw text — structured reading generation failed)*\n\n---\n\n{preview}{suffix}")

    # Handle duplicate persona IDs
    persona_id = persona["id"]
    persona_path = os.path.join(PERSONAS_DIR, f"{persona_id}.json")
    counter = 1
    while os.path.exists(persona_path):
        persona_id = f"{persona['id']}_{counter}"
        persona_path = os.path.join(PERSONAS_DIR, f"{persona_id}.json")
        counter += 1
    persona["id"] = persona_id

    # Save persona
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    with open(persona_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)

    return jsonify({
        "persona": persona,
        "reading_path": reading_path,
        "reading_quality": reading_quality,
    }), 201


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

if __name__ == "__main__":
    app.run(debug=True, port=5050)
