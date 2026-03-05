"""
claude_bridge.py — Claude Backend Bridge

Supports two backends, auto-detected at startup:
  1. Anthropic API  — if ANTHROPIC_API_KEY is set (works anywhere)
  2. Claude CLI      — subprocess with nesting/tool/prompt fixes

CLI fixes applied (from claude-cli-subprocess skill):
  1. Nesting guard  — unset CLAUDECODE env var before spawning
  2. Text-only gen  — pass --tools "" to disable tools
  3. Variadic flag   — deliver prompt via stdin, NOT positional arg
  4. Nested session  — explicit env dict, no shell=True
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Dict, Generator, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT = 3
TIMEOUT = 300  # seconds
API_MODEL = "claude-sonnet-4-20250514"

_semaphore = threading.Semaphore(MAX_CONCURRENT)
_active_count = 0
_active_lock = threading.Lock()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_use_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
_api_client = None

if _use_api:
    try:
        import anthropic
        _api_client = anthropic.Anthropic()
        logger.info("Claude backend: Anthropic API (model=%s)", API_MODEL)
    except ImportError:
        logger.warning("ANTHROPIC_API_KEY set but 'anthropic' package not installed — falling back to CLI")
        _use_api = False

if not _use_api:
    logger.info("Claude backend: CLI subprocess")

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _build_env() -> dict:
    """Build a clean environment for the Claude subprocess.

    * Strips CLAUDECODE to prevent recursive nesting.
    * Extends PATH so the `claude` binary is discoverable regardless of
      how the Flask server was launched (systemd, launchd, cron, etc.).
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # Fix 1: nesting guard

    extra_paths = [
        os.path.expanduser("~/.local/bin"),
        "/usr/local/bin",
        "/opt/homebrew/bin",
    ]
    env["PATH"] = ":".join(extra_paths) + ":" + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
) -> str:
    """Assemble a full prompt from system context, history, and new message.

    Format:
        <system>
        {system_prompt}
        </system>

        Human: ...
        Philosopher: ...
        Human: {user_message}
    """
    parts: list[str] = []

    if system_prompt:
        parts.append(f"<system>\n{system_prompt}\n</system>\n")

    for turn in conversation_history:
        role = turn.get("role", "human")
        content = turn.get("content", "")
        if role in ("human", "user"):
            parts.append(f"Human: {content}")
        else:
            parts.append(f"Philosopher: {content}")

    parts.append(f"Human: {user_message}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tracking helpers
# ---------------------------------------------------------------------------


def _inc_active() -> None:
    global _active_count
    with _active_lock:
        _active_count += 1


def _dec_active() -> None:
    global _active_count
    with _active_lock:
        _active_count = max(0, _active_count - 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stream_dialogue(
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
) -> Generator[str, None, None]:
    """Stream a dialogue response from Claude.

    Yields text chunks as they arrive. Uses the Anthropic API if available,
    otherwise falls back to the Claude CLI subprocess.

    Blocks on the concurrency semaphore if MAX_CONCURRENT processes are
    already running.
    """
    acquired = _semaphore.acquire(timeout=TIMEOUT)
    if not acquired:
        yield "[ERROR] Timed out waiting for an available Claude process slot."
        return

    _inc_active()

    try:
        if _use_api and _api_client:
            yield from _stream_dialogue_api(system_prompt, conversation_history, user_message)
        else:
            yield from _stream_dialogue_cli(system_prompt, conversation_history, user_message)
    finally:
        _dec_active()
        _semaphore.release()


def _stream_dialogue_api(
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
) -> Generator[str, None, None]:
    """Stream via Anthropic API."""
    messages = []
    for turn in conversation_history:
        role = turn.get("role", "human")
        content = turn.get("content", "")
        api_role = "user" if role in ("human", "user") else "assistant"
        messages.append({"role": api_role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        with _api_client.messages.stream(
            model=API_MODEL,
            max_tokens=4096,
            system=system_prompt or "",
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as exc:
        logger.exception("Error in API stream_dialogue")
        yield f"\n[ERROR] {exc}"


def _stream_dialogue_cli(
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
) -> Generator[str, None, None]:
    """Stream via Claude CLI subprocess."""
    full_prompt = _build_prompt(system_prompt, conversation_history, user_message)
    env = _build_env()
    proc: Optional[subprocess.Popen] = None

    try:
        proc = subprocess.Popen(
            ["claude", "--print", "--model", "opus", "--tools", ""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        proc.stdin.write(full_prompt)
        proc.stdin.close()

        deadline = time.monotonic() + TIMEOUT

        for line in proc.stdout:
            if time.monotonic() > deadline:
                proc.kill()
                yield "\n[ERROR] Claude process timed out."
                return
            yield line

        proc.wait(timeout=10)

        if proc.returncode != 0:
            stderr_output = proc.stderr.read() if proc.stderr else ""
            logger.error("Claude CLI exited with code %d: %s", proc.returncode, stderr_output)
            yield f"\n[ERROR] Claude CLI exited with code {proc.returncode}."

    except Exception as exc:
        logger.exception("Error in CLI stream_dialogue")
        yield f"\n[ERROR] {exc}"
        if proc and proc.poll() is None:
            proc.kill()


def query_claude(prompt: str) -> str:
    """Non-streaming single query. Returns the full response text.

    Useful for one-off tasks like generating a title slug, summarising
    a commitment, etc.
    """
    acquired = _semaphore.acquire(timeout=TIMEOUT)
    if not acquired:
        return "[ERROR] Timed out waiting for an available Claude process slot."

    _inc_active()
    try:
        if _use_api and _api_client:
            return _query_claude_api(prompt)
        return _query_claude_cli(prompt)
    finally:
        _dec_active()
        _semaphore.release()


def _query_claude_api(prompt: str) -> str:
    """Query via Anthropic API."""
    try:
        response = _api_client.messages.create(
            model=API_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.exception("Error in API query_claude")
        return f"[ERROR] {exc}"


def _query_claude_cli(prompt: str) -> str:
    """Query via Claude CLI subprocess."""
    env = _build_env()
    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "opus", "--tools", ""],
            input=prompt,
            capture_output=True,
            text=True,
            env=env,
            timeout=TIMEOUT,
        )
        if result.returncode != 0:
            logger.error("Claude CLI error (code %d): %s", result.returncode, result.stderr)
            return f"[ERROR] Claude CLI exited with code {result.returncode}."
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after %d seconds", TIMEOUT)
        return "[ERROR] Claude CLI timed out."
    except Exception as exc:
        logger.exception("Error in CLI query_claude")
        return f"[ERROR] {exc}"


def get_status() -> Dict:
    """Return bridge status for the /api/status endpoint."""
    with _active_lock:
        active = _active_count
    return {
        "active_count": active,
        "max_concurrent": MAX_CONCURRENT,
        "available_slots": MAX_CONCURRENT - active,
        "backend": "api" if (_use_api and _api_client) else "cli",
    }
