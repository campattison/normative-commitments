# Normative Commitments

A philosophical workbench for mapping your normative commitments, examining tensions between them, and pursuing reflective equilibrium through structured dialogue with AI philosopher personas.

## What It Does

- **3D Belief Graph** — interactive force-directed graph of your moral/philosophical commitments, with weighted edges showing support and tension relationships
- **Philosopher Dialogue** — engage in Socratic examination with AI personas modeled on specific philosophers (Korsgaard, Lazar, or create your own)
- **Automatic Belief Extraction** — the system identifies new beliefs and tensions from dialogue and proposes graph updates
- **PDF Persona Creation** — upload a philosopher's works to generate a custom dialogue persona

## Prerequisites

- Python 3.9+
- One of:
  - [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated, **or**
  - An [Anthropic API key](https://console.anthropic.com/)

## Quick Start

```bash
# Clone and install
git clone https://github.com/campattison/normative-commitments.git
cd normative-commitments
pip install -r server/requirements.txt

# Option A: Using Claude Code CLI (no extra config needed)
cd server && python app.py

# Option B: Using Anthropic API
export ANTHROPIC_API_KEY="sk-ant-..."
cd server && python app.py
```

Open `http://localhost:5050` in your browser.

The app auto-detects which backend is available. If `ANTHROPIC_API_KEY` is set, it uses the API; otherwise it uses the Claude CLI.

## How It Works

1. **Explore** the belief graph — see how your commitments relate, where tensions lie
2. **Examine** a belief by clicking a node and entering dialogue with a philosopher persona
3. **Extract** — during dialogue, the system identifies new beliefs and tensions to add to your graph
4. **Iterate** — add nodes and edges directly, upload readings, refine your positions

## Creating Personas

Philosopher personas are JSON files in `personas/`. Each defines a name, philosophical voice, key methods, and source texts.

To create a custom persona from a philosopher's works:
1. Upload a PDF via the web interface
2. The system extracts key themes and generates a persona definition
3. The new persona appears in the dialogue selector

## Architecture

```
server/
  app.py              # Flask backend — file API, dialogue streaming, belief extraction
  claude_bridge.py    # Claude backend bridge (API or CLI auto-detection)
  belief_extractor.py # Extracts beliefs and tensions from dialogue text
ui/
  index.html          # Single-page application
  app.js              # Frontend logic — graph rendering, dialogue, panels
  style.css           # Styling
personas/             # Philosopher persona definitions (JSON)
commitments/          # Your commitment files (created at runtime)
tensions/             # Identified tensions (created at runtime)
dialogues/            # Saved dialogue transcripts (created at runtime)
graph.json            # Your belief graph (created at runtime)
```

## Starting Fresh

Copy `graph.json.example` to `graph.json` to start with a minimal example graph, or just launch the app — it creates an empty graph automatically.

## License

MIT

## Acknowledgments

Built with [Three.js](https://threejs.org/), [3d-force-graph](https://github.com/vasturiano/3d-force-graph), [D3.js](https://d3js.org/), and [marked.js](https://marked.js.org/).
