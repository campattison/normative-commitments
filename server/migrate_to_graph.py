"""
migrate_to_graph.py — One-time migration from markdown cross-references to graph.json.

Reads commitment and tension files, extracts metadata, and produces the
initial graph structure (nodes + edges).

Usage:
    python3 server/migrate_to_graph.py
"""

from __future__ import annotations

import json
import os
import re
import sys

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
GRAPH_PATH = os.path.join(PROJECT_ROOT, "graph.json")

COMMITMENT_DIR = os.path.join(PROJECT_ROOT, "commitments")
TENSION_DIR = os.path.join(PROJECT_ROOT, "tensions")

# Tier assignments — advisory label only, not structural
TIER_MAP: dict[str, str] = {
    "normative_question": "metaethics",
    "metaethics": "metaethics",
    "consequentialism": "normative",
    "constructivism": "normative",
    "duties_and_claims": "normative",
    "political_philosophy": "applied",
    "practical_identity": "normative",
    "applied_positions": "applied",
}


def extract_confidence(content: str) -> str:
    """Extract confidence level from markdown content."""
    match = re.search(r"\*\*Confidence\*\*:\s*(.+)", content, re.IGNORECASE)
    if not match:
        match = re.search(r"Confidence:\s*(.+)", content, re.IGNORECASE)
    if not match:
        return "probable"
    val = match.group(1).strip().lower()
    if "certain" in val:
        return "certain"
    if "probable" in val:
        return "probable"
    if "tentative" in val:
        return "tentative"
    if "revision" in val:
        return "under-revision"
    return "probable"


def extract_title(content: str) -> str:
    """Extract the first heading as the label."""
    match = re.match(r"#\s+(.+)", content)
    if match:
        return match.group(1).strip()
    return ""


def build_graph() -> dict:
    """Build the full graph from existing markdown files."""
    nodes: list[dict] = []
    edges: list[dict] = []

    # --- Nodes from commitment files ---
    if os.path.isdir(COMMITMENT_DIR):
        for fname in sorted(os.listdir(COMMITMENT_DIR)):
            if not fname.endswith(".md"):
                continue
            slug = fname.replace(".md", "")
            fpath = os.path.join(COMMITMENT_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            label = extract_title(content) or slug.replace("_", " ").title()
            confidence = extract_confidence(content)
            tier = TIER_MAP.get(slug)

            nodes.append({
                "id": slug,
                "label": label,
                "type": "domain",
                "tier": tier,
                "confidence": confidence,
                "contentPath": f"commitments/{fname}",
                "parentDomain": None,
                "x": None,
                "y": None,
            })

    # --- Edges from tension files ---
    # Each tension file maps to a tension edge between two commitment nodes.
    tension_edge_map = {
        "util_vs_duty": {
            "source": "consequentialism",
            "target": "duties_and_claims",
            "description": (
                "If consequences are all that morally matters, duties to specific "
                "individuals can only be instrumentally justified — but the "
                "duties-and-claims commitment says the claim is not derivative "
                "from welfare calculations."
            ),
        },
        "constructivism_vs_realism": {
            "source": "constructivism",
            "target": "metaethics",
            "description": (
                "Constructivism says norms are built through rational reflection "
                "and don't exist independently, but several commitments seem to "
                "presuppose mind-independent moral truths."
            ),
        },
    }

    if os.path.isdir(TENSION_DIR):
        for fname in sorted(os.listdir(TENSION_DIR)):
            if not fname.endswith(".md"):
                continue
            slug = fname.replace(".md", "")
            mapping = tension_edge_map.get(slug)
            if not mapping:
                continue

            edges.append({
                "id": f"e-{slug}",
                "source": mapping["source"],
                "target": mapping["target"],
                "polarity": "tension",
                "weight": 0.5,
                "description": mapping["description"],
                "contentPath": f"tensions/{fname}",
            })

    # --- Additional edges inferred from cross-references in Tensions sections ---
    # From metaethics: links to constructivism_vs_realism → already covered
    # From consequentialism: links to util_vs_duty → already covered
    # From constructivism: links to constructivism_vs_realism → already covered
    # From duties_and_claims: links to util_vs_duty → already covered

    # Add a few support edges that reflect the philosophical structure
    # (these can be refined through dialogue — all start at 0.5 weight)
    support_edges = [
        {
            "id": "e-normq-metaethics-support",
            "source": "normative_question",
            "target": "metaethics",
            "polarity": "support",
            "weight": 0.5,
            "description": "The normative question frames the metaethical investigation.",
            "contentPath": None,
        },
        {
            "id": "e-normq-constructivism-support",
            "source": "normative_question",
            "target": "constructivism",
            "polarity": "support",
            "weight": 0.5,
            "description": "The normative question motivates the constructivist approach.",
            "contentPath": None,
        },
        {
            "id": "e-constructivism-practical-identity-support",
            "source": "constructivism",
            "target": "practical_identity",
            "polarity": "support",
            "weight": 0.5,
            "description": "Constructivism grounds norms in practical identity.",
            "contentPath": None,
        },
        {
            "id": "e-conseq-applied-support",
            "source": "consequentialism",
            "target": "applied_positions",
            "polarity": "support",
            "weight": 0.5,
            "description": "Consequentialist reasoning shapes the applied positions (especially AI ethics and climate).",
            "contentPath": None,
        },
        {
            "id": "e-duties-political-support",
            "source": "duties_and_claims",
            "target": "political_philosophy",
            "polarity": "support",
            "weight": 0.5,
            "description": "Duties to particular others motivate contractualist political philosophy.",
            "contentPath": None,
        },
    ]
    edges.extend(support_edges)

    return {
        "version": 1,
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    graph = build_graph()

    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"Graph written to {GRAPH_PATH}")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")

    # Validate: no duplicate IDs
    node_ids = [n["id"] for n in graph["nodes"]]
    edge_ids = [e["id"] for e in graph["edges"]]
    assert len(node_ids) == len(set(node_ids)), "Duplicate node IDs found!"
    assert len(edge_ids) == len(set(edge_ids)), "Duplicate edge IDs found!"

    # Validate: all edge sources/targets reference existing nodes
    node_set = set(node_ids)
    for edge in graph["edges"]:
        assert edge["source"] in node_set, f"Edge {edge['id']} references unknown source: {edge['source']}"
        assert edge["target"] in node_set, f"Edge {edge['id']} references unknown target: {edge['target']}"

    print("  Validation: OK (no duplicates, all references valid)")


if __name__ == "__main__":
    main()
