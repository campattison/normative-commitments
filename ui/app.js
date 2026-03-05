/* ==========================================================================
   Normative Commitments — Client Application
   3D belief graph (Three.js / 3d-force-graph), SSE dialogue
   ========================================================================== */

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // Constants
  // -----------------------------------------------------------------------

  const API_BASE = '/api';
  const app = document.getElementById('app');

  const TIER_COLORS = {
    metaethics: '#4ecca3',
    normative:  '#a855f7',
    applied:    '#3b82f6',
  };
  const DEFAULT_NODE_COLOR = '#8892a8';

  const EDGE_COLORS = {
    support: '#4ecca3',
    tension: '#e94560',
  };

  const DISK_RADIUS = 400;        // Poincaré disk radius in graph units

  // Confidence visual encoding
  const CONFIDENCE_OPACITY = {
    certain:          1.0,
    probable:         0.85,
    tentative:        0.6,
    'under-revision': 0.5,
    retracted:        0.2,
  };
  const CONFIDENCE_RING = {
    certain:          { width: 3,   dash: null },
    probable:         { width: 2,   dash: null },
    tentative:        { width: 1.5, dash: [4, 3] },
    'under-revision': { width: 2,   dash: [6, 3] },
    retracted:        { width: 1,   dash: [2, 4] },
  };

  const BEZIER_CURVATURE = 0.15;

  // --- Trumpet geometry parameters (module scope for link callbacks) ---
  const TRUMPET_TIP_RADIUS = 15;      // XY radius at the narrow end (foundational)
  const TRUMPET_BELL_RADIUS = 200;    // XY radius at the wide end (peripheral)
  const TRUMPET_LENGTH = DISK_RADIUS; // Z extent of the trumpet

  // --- Dynamic trumpet scaling ---
  const TRUMPET_BASE_NODE_COUNT = 6;
  let trumpetScale = 1;

  function computeTrumpetScale(nodeCount) {
    return Math.max(1, Math.sqrt(nodeCount / TRUMPET_BASE_NODE_COUNT));
  }
  let _tipRadiusComputing = false; // re-entrancy guard
  function tipRadius() {
    const base = TRUMPET_TIP_RADIUS * trumpetScale;
    if (_tipRadiusComputing || !graphData) return base;
    _tipRadiusComputing = true;
    try {
      // Count nodes near the tip (depth < 0.15 normalized)
      const depths = getComputedDepths();
      const degrees = computeDegrees();
      let tipNodeCount = 0;
      let maxTipDiameter = 0;
      (graphData.nodes || []).forEach(n => {
        const d = depths.get(n.id) ?? 0.5;
        if (d < 0.15) {
          tipNodeCount++;
          const diam = 2 * nodeRadius(n, degrees);
          if (diam > maxTipDiameter) maxTipDiameter = diam;
        }
      });
      if (tipNodeCount <= 1) return base;
      // Each node needs ~1.3x its diameter in arc length (tight but no overlap)
      const minCircumference = tipNodeCount * maxTipDiameter * 1.3;
      const minFromPacking = minCircumference / (2 * Math.PI);
      // Cap at 2x base to preserve trumpet flare ratio
      return Math.min(base * 2, Math.max(base, minFromPacking));
    } finally {
      _tipRadiusComputing = false;
    }
  }
  function bellRadius() { return TRUMPET_BELL_RADIUS * trumpetScale; }
  function trumpetLen() { return TRUMPET_LENGTH * trumpetScale; }
  function diskRadius() { return DISK_RADIUS * trumpetScale; }

  // Trumpet radius at a given z (exponential flare)
  function trumpetRadius(z) {
    const t = Math.min(1, Math.max(0, Math.abs(z) / trumpetLen()));
    return tipRadius() + (bellRadius() - tipRadius()) * (Math.exp(2.5 * t) - 1) / (Math.exp(2.5) - 1);
  }

  // Interpolate a point on the trumpet surface between two surface points
  // Returns {x, y, z} for parameter t ∈ [0, 1]
  function trumpetSurfacePoint(start, end, t) {
    // Interpolate z linearly
    const z = start.z + t * (end.z - start.z);
    // Interpolate angle (shortest arc)
    const startTheta = Math.atan2(start.y, start.x);
    const endTheta = Math.atan2(end.y, end.x);
    let dTheta = endTheta - startTheta;
    if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    if (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    const theta = startTheta + t * dTheta;
    // Radius at this z
    const r = trumpetRadius(z);
    return { x: r * Math.cos(theta), y: r * Math.sin(theta), z };
  }

  // Cached degree map — invalidated when graph changes
  let _cachedDegrees = null;
  let _cachedDegreesKey = null;

  // Cached depth map (computed from support-edge DAG) — invalidated with degrees
  let _cachedDepths = null;
  let _cachedDepthsKey = null;

  // Cached transitive support counts — invalidated with degrees
  let _cachedTransitiveCounts = null;
  let _cachedTransitiveCountsKey = null;

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------

  let filesCache = null;
  let dialogueHistory = [];
  let dialogueContext = null;
  let dialogueContextContent = null;
  let isStreaming = false;

  // Typewriter state — buffers streamed text and reveals it gradually
  let typewriterQueue = '';      // characters waiting to be revealed
  let typewriterRevealed = '';   // characters already shown
  let typewriterRafId = null;    // animation frame handle
  const CHARS_PER_FRAME = 2;    // characters revealed per animation frame (~120 chars/sec at 60fps)

  // Smart autoscroll — only auto-scroll when user is near bottom
  let userHasScrolledUp = false;
  const SCROLL_THRESHOLD = 100;

  function isNearBottom(el) {
    return (el.scrollHeight - el.scrollTop - el.clientHeight) < SCROLL_THRESHOLD;
  }

  // Contest state
  let contestNode = null;
  let contestSubgraph = null;
  let contestMode = false;

  // Graph state
  let graphData = null;
  let simulation = null;
  let selectedNode = null;
  let selectedEdge = null;
  let domainScope = null;
  let addEdgeMode = false;
  let addEdgeSource = null;

  // Persona state
  let personasCache = null;
  let activePersonaId = 'korsgaard';
  let activePersonaData = null;
  let personaDetailExpanded = false;

  // 3D graph state
  let graph3d = null;        // ForceGraph3D instance
  let hoveredNode = null;    // node under cursor

  // Maps from ID → Three.js Group for highlight updates (avoids library internals)
  const _nodeThreeMap = new Map();  // node.id → THREE.Group
  const _linkThreeMap = new Map();  // link.id → THREE.Group

  // -----------------------------------------------------------------------
  // Routing
  // -----------------------------------------------------------------------

  const routes = {
    'graph':      renderGraph,
    'commitment': selectNodeOnGraph,
    'dialogue':   renderDialogue,
    'readings':   renderReadings,
    'reading':    renderReading,
  };

  function handleRoute() {
    const hash = window.location.hash.slice(1) || 'graph';
    const [route, ...paramParts] = hash.split('/');
    const param = paramParts.join('/');
    const baseRoute = route.split('?')[0];

    if (baseRoute === 'dashboard' || baseRoute === 'tensions' || baseRoute === 'tension') {
      window.location.hash = 'graph';
      return;
    }

    if (baseRoute === 'graph') {
      app.classList.add('graph-active');
    } else {
      app.classList.remove('graph-active');
      stopSimulation();
      cancelAnimFrame();
    }

    updateNavActive(baseRoute);
    const handler = routes[baseRoute];
    if (handler) {
      handler(param);
    } else {
      renderNotFound();
    }
  }

  function selectNodeOnGraph(name) {
    window.location.hash = 'graph';
    setTimeout(() => {
      if (graphData) {
        const node = graphData.nodes.find(n => n.id === name);
        if (node) {
          selectedNode = node;
          selectedEdge = null;
          renderDetailPanel();
          updateHighlight();
        }
      }
    }, 300);
  }

  function updateNavActive(route) {
    document.querySelectorAll('.nav-link').forEach(link => {
      link.classList.toggle('active', link.getAttribute('data-route') === route);
    });
  }

  function navigate(hash) {
    window.location.hash = hash;
  }

  window.addEventListener('hashchange', handleRoute);
  window.addEventListener('load', handleRoute);

  // -----------------------------------------------------------------------
  // Mobile nav toggle
  // -----------------------------------------------------------------------

  const navToggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
    navLinks.addEventListener('click', (e) => {
      if (e.target.classList.contains('nav-link')) navLinks.classList.remove('open');
    });
  }

  // -----------------------------------------------------------------------
  // API Functions
  // -----------------------------------------------------------------------

  async function fetchFiles() {
    if (filesCache) return filesCache;
    const r = await fetch(`${API_BASE}/files`);
    if (!r.ok) throw new Error(`Fetch files failed: ${r.status}`);
    filesCache = await r.json();
    return filesCache;
  }

  function invalidateCache() { filesCache = null; }

  async function fetchFile(path) {
    const r = await fetch(`${API_BASE}/files/${path}`);
    if (!r.ok) throw new Error(`Fetch file failed: ${r.status}`);
    return r.json();
  }

  async function saveFile(path, content) {
    const r = await fetch(`${API_BASE}/files/${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
    if (!r.ok) throw new Error(`Save failed: ${r.status}`);
    invalidateCache();
    return r.json();
  }

  async function fetchGraph() {
    const r = await fetch(`${API_BASE}/graph`);
    if (!r.ok) throw new Error(`Fetch graph failed: ${r.status}`);
    return r.json();
  }

  async function patchNode(data) {
    const r = await fetch(`${API_BASE}/graph/node`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!r.ok) throw new Error(`Patch node failed: ${r.status}`);
    return r.json();
  }

  async function deleteNodeApi(id) {
    const r = await fetch(`${API_BASE}/graph/node/${id}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`Delete node failed: ${r.status}`);
    return r.json();
  }

  async function patchEdge(data) {
    const r = await fetch(`${API_BASE}/graph/edge`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!r.ok) throw new Error(`Patch edge failed: ${r.status}`);
    return r.json();
  }

  async function deleteEdgeApi(id) {
    const r = await fetch(`${API_BASE}/graph/edge/${id}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`Delete edge failed: ${r.status}`);
    return r.json();
  }

  async function sendDialogueRequest(systemPrompt, history, message) {
    const r = await fetch(`${API_BASE}/dialogue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system_prompt: systemPrompt, history, message }),
    });
    if (!r.ok) throw new Error(`Dialogue failed: ${r.status}`);
    return r;
  }

  async function saveDialogue(topic, messages) {
    const r = await fetch(`${API_BASE}/dialogue/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, messages }),
    });
    if (!r.ok) throw new Error(`Save dialogue failed: ${r.status}`);
    invalidateCache();
    return r.json();
  }

  // -----------------------------------------------------------------------
  // Persona API
  // -----------------------------------------------------------------------

  async function fetchPersonas() {
    if (personasCache) return personasCache;
    const r = await fetch(`${API_BASE}/personas`);
    if (!r.ok) throw new Error(`Fetch personas failed: ${r.status}`);
    personasCache = await r.json();
    return personasCache;
  }

  function invalidatePersonasCache() { personasCache = null; }

  async function fetchPersona(id) {
    const r = await fetch(`${API_BASE}/personas/${id}`);
    if (!r.ok) throw new Error(`Fetch persona failed: ${r.status}`);
    return r.json();
  }

  async function deletePersona(id) {
    const r = await fetch(`${API_BASE}/personas/${id}`, { method: 'DELETE' });
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      throw new Error(data.error || `Delete persona failed: ${r.status}`);
    }
    invalidatePersonasCache();
    return r.json();
  }

  async function uploadPdfForPersona(file) {
    const form = new FormData();
    form.append('pdf', file);
    const r = await fetch(`${API_BASE}/personas/generate`, { method: 'POST', body: form });
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      throw new Error(data.error || `Upload failed: ${r.status}`);
    }
    invalidatePersonasCache();
    invalidateCache();
    return r.json();
  }

  async function ensureActivePersona() {
    if (activePersonaData && activePersonaData.id === activePersonaId) return;
    try {
      activePersonaData = await fetchPersona(activePersonaId);
    } catch (err) {
      console.warn(`Failed to load persona "${activePersonaId}", falling back to korsgaard:`, err);
      activePersonaId = 'korsgaard';
      try {
        activePersonaData = await fetchPersona('korsgaard');
      } catch (fallbackErr) {
        console.error('Failed to load fallback persona:', fallbackErr);
        activePersonaData = null;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  function slugify(t) { return t.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, ''); }
  function titleCase(s) { return s.replace(/[-_]/g, ' ').replace(/\.md$/i, '').replace(/\b\w/g, c => c.toUpperCase()); }
  function truncate(t, n) { if (!t || t.length <= n) return t || ''; return t.slice(0, n).replace(/\s+\S*$/, '') + '\u2026'; }
  function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
  function setContent(h) { app.innerHTML = h; }
  function showLoading(m) { setContent(`<div class="loading">${escapeHtml(m || 'Loading...')}</div>`); }
  function showError(m) { setContent(`<div class="error-message">${escapeHtml(m)}</div>`); }
  function filterFilesByDir(f, d) { return f.filter(x => (x.path || x).startsWith(d + '/') && (x.path || x).endsWith('.md')); }
  function fileBasename(p) { return p.split('/').pop().replace(/\.md$/i, ''); }

  function renderMarkdown(md) {
    if (typeof marked !== 'undefined' && marked.parse) return marked.parse(md);
    return `<pre>${escapeHtml(md)}</pre>`;
  }

  function confidenceBadge(c) {
    if (!c) return '';
    return `<span class="badge badge-${c}">${c.replace('-', ' ')}</span>`;
  }

  function tierBadge(tier) {
    if (!tier) return '';
    const color = TIER_COLORS[tier] || DEFAULT_NODE_COLOR;
    return `<span class="badge badge-tier" style="background:rgba(${hexToRGB(color)},0.15);color:${color};border:1px solid rgba(${hexToRGB(color)},0.3)">${tier}</span>`;
  }

  function hexToRGB(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `${r},${g},${b}`;
  }

  function hexToRGBA(hex, a) {
    return `rgba(${hexToRGB(hex)},${a})`;
  }

  function upsertNode(node) {
    const i = graphData.nodes.findIndex(n => n.id === node.id);
    if (i >= 0) graphData.nodes[i] = node;
    else graphData.nodes.push(node);
  }

  function upsertEdge(edge) {
    const i = graphData.edges.findIndex(e => e.id === edge.id);
    if (i >= 0) graphData.edges[i] = edge;
    else graphData.edges.push(edge);
  }

  // -----------------------------------------------------------------------
  // Node sizing — by degree
  // -----------------------------------------------------------------------

  function computeDegrees() {
    // Cache by node+edge count to avoid recomputing every frame
    const key = `${graphData.nodes.length}:${graphData.edges.length}`;
    if (_cachedDegrees && _cachedDegreesKey === key) return _cachedDegrees;

    const deg = {};
    graphData.nodes.forEach(n => { deg[n.id] = 0; });
    graphData.edges.forEach(e => {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      if (deg[s] !== undefined) deg[s]++;
      if (deg[t] !== undefined) deg[t]++;
    });
    _cachedDegrees = deg;
    _cachedDegreesKey = key;
    return deg;
  }

  function invalidateDegreeCache() {
    _cachedDegrees = null;
    _cachedDegreesKey = null;
    _cachedDepths = null;
    _cachedDepthsKey = null;
    _cachedTransitiveCounts = null;
    _cachedTransitiveCountsKey = null;
  }

  function nodeRadius(node, degrees) {
    const maxDeg = Math.max(...Object.values(degrees), 1);
    const d = degrees[node.id] || 0;
    const base = 16 + (d / maxDeg) * 20;
    if (node.confidence === 'retracted') return base * 0.6;
    return base;
  }

  function nodeColor(node) {
    return TIER_COLORS[node.tier] || DEFAULT_NODE_COLOR;
  }

  function truncateLabel(label, maxLen) {
    if (!label || label.length <= maxLen) return label;
    return label.slice(0, maxLen - 1) + '\u2026';
  }

  // -----------------------------------------------------------------------
  // Bezier helpers
  // -----------------------------------------------------------------------

  function bezierControlPoint(sx, sy, tx, ty, offset) {
    const mx = (sx + tx) / 2;
    const my = (sy + ty) / 2;
    const dx = tx - sx;
    const dy = ty - sy;
    const k = BEZIER_CURVATURE + (offset || 0);
    return { x: mx - dy * k, y: my + dx * k };
  }

  function bezierPoint(p0x, p0y, cpx, cpy, p2x, p2y, t) {
    const mt = 1 - t;
    return {
      x: mt * mt * p0x + 2 * mt * t * cpx + t * t * p2x,
      y: mt * mt * p0y + 2 * mt * t * cpy + t * t * p2y,
    };
  }

  function bezierTangent(p0x, p0y, cpx, cpy, p2x, p2y, t) {
    const mt = 1 - t;
    return {
      x: 2 * mt * (cpx - p0x) + 2 * t * (p2x - cpx),
      y: 2 * mt * (cpy - p0y) + 2 * t * (p2y - cpy),
    };
  }

  function distToBezier(px, py, p0x, p0y, cpx, cpy, p2x, p2y) {
    let minDist = Infinity;
    for (let t = 0; t <= 1; t += 0.05) {
      const pt = bezierPoint(p0x, p0y, cpx, cpy, p2x, p2y, t);
      const dx = px - pt.x;
      const dy = py - pt.y;
      const d = dx * dx + dy * dy;
      if (d < minDist) minDist = d;
    }
    return Math.sqrt(minDist);
  }

  // -----------------------------------------------------------------------
  // Canvas — coordinate transforms
  // -----------------------------------------------------------------------

  function screenToGraph(sx, sy) {
    return {
      x: (sx - transform.x) / transform.scale,
      y: (sy - transform.y) / transform.scale,
    };
  }

  function findNodeAt(gx, gy) {
    const degrees = computeDegrees();
    for (let i = graphData.nodes.length - 1; i >= 0; i--) {
      const n = graphData.nodes[i];
      if (n.x == null || n.y == null) continue;
      const r = nodeRadius(n, degrees);
      const dx = gx - n.x;
      const dy = gy - n.y;
      if (dx * dx + dy * dy <= r * r) return n;
    }
    return null;
  }

  function findEdgeAt(gx, gy) {
    // Build parallel-edge offset map (must match drawGraph logic)
    const pairCount = {};
    const pairIndex = {};
    for (const e of graphData.edges) {
      const sId = typeof e.source === 'object' ? e.source.id : e.source;
      const tId = typeof e.target === 'object' ? e.target.id : e.target;
      const pk = [sId, tId].sort().join('::');
      pairCount[pk] = (pairCount[pk] || 0) + 1;
      pairIndex[e.id] = pairCount[pk] - 1;
    }

    for (const e of graphData.edges) {
      const src = typeof e.source === 'object' ? e.source : graphData.nodes.find(n => n.id === e.source);
      const tgt = typeof e.target === 'object' ? e.target : graphData.nodes.find(n => n.id === e.target);
      if (!src || !tgt || src.x == null || tgt.x == null) continue;

      const sId = src.id || (typeof e.source === 'string' ? e.source : e.source.id);
      const tId = tgt.id || (typeof e.target === 'string' ? e.target : e.target.id);
      const pk = [sId, tId].sort().join('::');
      const total = pairCount[pk] || 1;
      const idx = pairIndex[e.id] || 0;
      const curveOffset = total > 1 ? (idx - (total - 1) / 2) * 0.12 : 0;

      const cp = bezierControlPoint(src.x, src.y, tgt.x, tgt.y, curveOffset);
      const dist = distToBezier(gx, gy, src.x, src.y, cp.x, cp.y, tgt.x, tgt.y);
      if (dist < 10) return e;
    }
    return null;
  }

  // -----------------------------------------------------------------------
  // Graph — simulation helpers
  // -----------------------------------------------------------------------

  function stopSimulation() {
    if (simulation) { simulation.stop(); simulation = null; }
  }

  function cancelAnimFrame() {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  }

  // -----------------------------------------------------------------------
  // Graph View
  // -----------------------------------------------------------------------

  async function renderGraph() {
    document.removeEventListener('keydown', handleGraphKeydown);

    if (!graphData) {
      app.innerHTML = '<div class="graph-layout"><div class="graph-canvas-container"><div class="loading">Loading graph...</div></div></div>';
      try {
        graphData = await fetchGraph();
      } catch {
        app.innerHTML = '<div class="graph-layout"><div class="graph-canvas-container"><div class="error-message">Could not load graph.</div></div></div>';
        return;
      }
    }

    let html = '<div class="graph-layout">';
    html += '<div class="graph-canvas-container" id="graph-container">';

    // Breadcrumb
    if (domainScope) {
      const sn = graphData.nodes.find(n => n.id === domainScope);
      html += `<div class="graph-breadcrumb">
        <a href="#" id="breadcrumb-back">All</a>
        <span class="separator">&rsaquo;</span>
        <span class="current">${escapeHtml(sn ? sn.label : domainScope)}</span>
      </div>`;
    }

    // Empty state
    if (graphData.nodes.length === 0) {
      html += `<div class="graph-empty-state">
        <h3>Your belief graph is empty</h3>
        <p>Start a dialogue and capture insights to build your normative landscape, or add beliefs directly.</p>
      </div>`;
    }

    // Legend
    html += `<div class="graph-legend">
      <div class="graph-legend-title">Tiers</div>
      <div class="legend-item"><span class="legend-dot" style="background:${TIER_COLORS.metaethics}"></span> Metaethics</div>
      <div class="legend-item"><span class="legend-dot" style="background:${TIER_COLORS.normative}"></span> Normative</div>
      <div class="legend-item"><span class="legend-dot" style="background:${TIER_COLORS.applied}"></span> Applied</div>
      <div class="legend-item"><span class="legend-dot" style="background:${DEFAULT_NODE_COLOR}"></span> Unassigned</div>
      <div class="legend-divider"></div>
      <div class="graph-legend-title">Edges</div>
      <div class="legend-item"><span class="legend-line" style="background:${EDGE_COLORS.support}"></span> Support</div>
      <div class="legend-item"><span class="legend-line dashed" style="border-color:${EDGE_COLORS.tension}"></span> Tension</div>
    </div>`;

    // Toolbar
    html += `<div class="graph-toolbar">
      <button class="btn btn-small" id="btn-add-node">+ Add Belief</button>
      <button class="btn btn-small" id="btn-zoom-fit">Fit</button>
    </div>`;

    // Stats
    if (graphData.nodes.length > 0) {
      const supportCount = graphData.edges.filter(e => e.polarity === 'support').length;
      const tensionCount = graphData.edges.filter(e => e.polarity === 'tension').length;
      html += `<div class="graph-stats">
        <span><span class="stat-value">${graphData.nodes.length}</span> beliefs</span>
        <span><span class="stat-value">${supportCount}</span> supports</span>
        <span><span class="stat-value">${tensionCount}</span> tensions</span>
      </div>`;
    }

    html += '</div>'; // close canvas container
    html += '<div id="detail-panel-container"></div>';
    html += '</div>'; // close layout

    app.innerHTML = html;

    initCanvas();

    document.getElementById('btn-add-node')?.addEventListener('click', showAddNodeModal);
    document.getElementById('btn-zoom-fit')?.addEventListener('click', zoomToFit);
    document.getElementById('breadcrumb-back')?.addEventListener('click', (e) => {
      e.preventDefault();
      exitDomainScope();
    });

    document.addEventListener('keydown', handleGraphKeydown);

    if (selectedNode || selectedEdge) renderDetailPanel();
  }

  function handleGraphKeydown(e) {
    if (e.key === 'Escape') {
      if (addEdgeMode) { addEdgeMode = false; addEdgeSource = null; updateCanvasCursor(); return; }
      if (domainScope) { exitDomainScope(); return; }
      if (selectedNode || selectedEdge) {
        selectedNode = null; selectedEdge = null;
        renderDetailPanel(); updateHighlight();
        return;
      }
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedEdge) {
      const ae = document.activeElement;
      if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.tagName === 'SELECT')) return;
      handleDeleteEdge(selectedEdge.id);
    }
  }

  // -----------------------------------------------------------------------
  // 3D Graph — initialization
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  // Dynamic depth from support-edge DAG
  // -----------------------------------------------------------------------

  /**
   * Compute depth [0, 1] for every node from the support-edge DAG.
   * Roots (no incoming support but at least one outgoing) get depth 0.
   * Each child gets max(current, parent + 1) — longest path from any root.
   * Depths are normalized to [0, 1]. Isolated nodes (no support edges) → 0.5.
   */
  function computeDepthsFromDAG(gData) {
    const nodes = gData.nodes || [];
    const edges = (gData.edges || gData.links || []);

    // Build adjacency from support edges only (source grounds target)
    const children = {};   // source → [target, ...]
    const inDegree = {};   // target → count of incoming support edges
    const nodeIds = new Set();

    nodes.forEach(n => {
      const id = typeof n === 'object' ? (n.id || n) : n;
      nodeIds.add(id);
      children[id] = [];
      inDegree[id] = 0;
    });

    edges.forEach(e => {
      if ((e.polarity || 'support') !== 'support') return;
      const src = typeof e.source === 'object' ? e.source.id : e.source;
      const tgt = typeof e.target === 'object' ? e.target.id : e.target;
      if (!nodeIds.has(src) || !nodeIds.has(tgt)) return;
      children[src].push(tgt);
      inDegree[tgt] = (inDegree[tgt] || 0) + 1;
    });

    // Identify roots: no incoming support edges, but at least one outgoing
    const queue = [];
    const depth = {};
    nodeIds.forEach(id => {
      depth[id] = -1; // unvisited
      if ((inDegree[id] || 0) === 0 && children[id].length > 0) {
        queue.push(id);
        depth[id] = 0;
      }
    });

    // Kahn's BFS — longest path via max(current, parent + 1)
    const maxEnqueues = nodes.length; // cycle safety cap
    const enqueueCount = {};
    let head = 0;
    while (head < queue.length) {
      const curr = queue[head++];
      for (const child of children[curr]) {
        const newDepth = depth[curr] + 1;
        if (newDepth > (depth[child] === -1 ? -Infinity : depth[child])) {
          depth[child] = newDepth;
          enqueueCount[child] = (enqueueCount[child] || 0) + 1;
          if (enqueueCount[child] <= maxEnqueues) {
            queue.push(child);
          }
        }
      }
    }

    // Find max layer for normalization
    let maxLayer = 0;
    nodeIds.forEach(id => {
      if (depth[id] > maxLayer) maxLayer = depth[id];
    });

    // Build result map: normalize to [0, 1] with sub-layer offsets based on
    // transitive support counts. Nodes with more descendants sit closer to
    // the tip center within their layer band.
    const tc = computeTransitiveSupportCounts(gData);
    let maxTc = 0;
    nodeIds.forEach(id => { if ((tc[id] || 0) > maxTc) maxTc = tc[id]; });

    const result = new Map();
    nodeIds.forEach(id => {
      if (depth[id] < 0) {
        // Isolated from support edges
        result.set(id, 0.5);
      } else if (maxLayer === 0) {
        // Single layer — use transitive count for sub-layer spread
        const frac = maxTc > 0 ? (1 - (tc[id] || 0) / maxTc) : 0;
        result.set(id, frac * 0.8);
      } else {
        const layer = depth[id];
        const bandWidth = 1 / (maxLayer + 1);
        const baseDepth = layer * bandWidth;
        const subLayerOffset = maxTc > 0
          ? (1 - (tc[id] || 0) / maxTc) * 0.8 * bandWidth
          : 0;
        result.set(id, baseDepth + subLayerOffset);
      }
    });

    return result;
  }

  function getComputedDepths() {
    if (!graphData) return new Map();
    const key = `${(graphData.nodes || []).length}:${(graphData.edges || []).length}`;
    if (_cachedDepths && _cachedDepthsKey === key) return _cachedDepths;
    _cachedDepths = computeDepthsFromDAG(graphData);
    _cachedDepthsKey = key;
    return _cachedDepths;
  }

  /**
   * Compute transitive support counts: for each node, how many distinct
   * descendants are reachable via outgoing support edges (DFS).
   */
  function computeTransitiveSupportCounts(gData) {
    const nodes = gData.nodes || [];
    const edges = (gData.edges || gData.links || []);
    const nodeIds = new Set();
    const children = {};

    nodes.forEach(n => {
      const id = typeof n === 'object' ? (n.id || n) : n;
      nodeIds.add(id);
      children[id] = [];
    });

    edges.forEach(e => {
      if ((e.polarity || 'support') !== 'support') return;
      const src = typeof e.source === 'object' ? e.source.id : e.source;
      const tgt = typeof e.target === 'object' ? e.target.id : e.target;
      if (!nodeIds.has(src) || !nodeIds.has(tgt)) return;
      children[src].push(tgt);
    });

    const counts = {};
    const memo = {};

    function dfs(id) {
      if (memo[id] !== undefined) return memo[id];
      const reachable = new Set();
      memo[id] = reachable; // set early to handle cycles
      for (const child of (children[id] || [])) {
        reachable.add(child);
        for (const desc of dfs(child)) {
          reachable.add(desc);
        }
      }
      return reachable;
    }

    nodeIds.forEach(id => {
      counts[id] = dfs(id).size;
    });

    return counts;
  }

  function getTransitiveSupportCounts() {
    if (!graphData) return {};
    const key = `${(graphData.nodes || []).length}:${(graphData.edges || []).length}`;
    if (_cachedTransitiveCounts && _cachedTransitiveCountsKey === key) return _cachedTransitiveCounts;
    _cachedTransitiveCounts = computeTransitiveSupportCounts(graphData);
    _cachedTransitiveCountsKey = key;
    return _cachedTransitiveCounts;
  }

  // Hyperbolic depth → Z coordinate (foundational at center/top, peripheral below)
  function nodeDepthZ(node) {
    const depth = getComputedDepths().get(node.id) ?? 0.5;
    // Exponential: foundational nodes near z=0 (top), peripheral fall away
    const expDenom = 1 - Math.exp(-2.5);
    return -diskRadius() * (1 - Math.exp(-2.5 * depth)) / expDenom;
  }

  function initCanvas() {
    const container = document.getElementById('graph-container');
    if (!container || !graphData) return;

    // Clean up previous instance
    if (graph3d) {
      graph3d._destructor && graph3d._destructor();
      graph3d = null;
    }
    container.querySelectorAll('canvas, div.scene-container').forEach(el => el.remove());

    if (graphData.nodes.length === 0) return;

    invalidateDegreeCache();
    const degrees = computeDegrees();

    // Recompute trumpet scale based on current node count
    trumpetScale = computeTrumpetScale(graphData.nodes.length);

    // Build data for 3d-force-graph
    const nodes3d = graphData.nodes.map(n => ({
      ...n,
      _color: nodeColor(n),
      _radius: nodeRadius(n, degrees),
      _opacity: CONFIDENCE_OPACITY[n.confidence] || 0.7,
    }));

    const links3d = graphData.edges.map(e => ({
      ...e,
      source: typeof e.source === 'object' ? e.source.id : e.source,
      target: typeof e.target === 'object' ? e.target.id : e.target,
    }));

    const gData = { nodes: nodes3d, links: links3d };

    // Hex color string → int
    function colorToInt(hex) {
      return parseInt(hex.replace('#', ''), 16);
    }

    graph3d = ForceGraph3D()(container)
      .graphData(gData)
      .backgroundColor('#0f1117')
      .width(container.clientWidth)
      .height(container.clientHeight)
      // --- Nodes ---
      .nodeThreeObject(node => {
        const group = new THREE.Group();

        // Sphere
        const r = node._radius * 0.35;
        const geo = new THREE.SphereGeometry(r, 24, 16);
        const mat = new THREE.MeshLambertMaterial({
          color: colorToInt(node._color),
          transparent: true,
          opacity: node._opacity,
        });
        const sphere = new THREE.Mesh(geo, mat);
        sphere.userData._origOpacity = node._opacity;
        group.add(sphere);

        // Confidence ring
        const ring = CONFIDENCE_RING[node.confidence] || CONFIDENCE_RING.probable;
        const ringGeo = new THREE.RingGeometry(r + 0.5, r + 0.5 + ring.width * 0.3, 48);
        const ringOpacity = ring.dash ? 0.4 : 0.7;
        const ringMat = new THREE.MeshBasicMaterial({
          color: colorToInt(node._color),
          transparent: true,
          opacity: ringOpacity,
          side: THREE.DoubleSide,
        });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.userData._origOpacity = ringOpacity;
        group.add(ringMesh);

        // Text label
        const canvas2d = document.createElement('canvas');
        const ctx2d = canvas2d.getContext('2d');
        const label = truncateLabel(node.label, 40);
        const fontSize = 28;
        ctx2d.font = `${fontSize}px Inter, Helvetica Neue, Arial, sans-serif`;
        const textWidth = ctx2d.measureText(label).width;
        canvas2d.width = textWidth + 16;
        canvas2d.height = fontSize + 12;
        ctx2d.font = `${fontSize}px Inter, Helvetica Neue, Arial, sans-serif`;
        ctx2d.fillStyle = '#e0e0e6';
        ctx2d.textAlign = 'center';
        ctx2d.textBaseline = 'middle';
        ctx2d.fillText(label, canvas2d.width / 2, canvas2d.height / 2);

        const texture = new THREE.CanvasTexture(canvas2d);
        texture.minFilter = THREE.LinearFilter;
        const spriteMat = new THREE.SpriteMaterial({
          map: texture,
          transparent: true,
          opacity: 0.85,
          depthTest: false,
        });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.userData._origOpacity = 0.85;
        const spriteScale = canvas2d.width / canvas2d.height;
        sprite.scale.set(spriteScale * 4, 4, 1);
        sprite.position.y = -(r + 4);
        group.add(sprite);

        // Confidence sub-label
        if (node.confidence && node.confidence !== 'retracted') {
          const c2 = document.createElement('canvas');
          const cx2 = c2.getContext('2d');
          const confText = node.confidence;
          const fs2 = 20;
          cx2.font = `${fs2}px Inter, Helvetica Neue, Arial, sans-serif`;
          const tw2 = cx2.measureText(confText).width;
          c2.width = tw2 + 12;
          c2.height = fs2 + 8;
          cx2.font = `${fs2}px Inter, Helvetica Neue, Arial, sans-serif`;
          cx2.fillStyle = '#8892a8';
          cx2.textAlign = 'center';
          cx2.textBaseline = 'middle';
          cx2.fillText(confText, c2.width / 2, c2.height / 2);
          const tex2 = new THREE.CanvasTexture(c2);
          tex2.minFilter = THREE.LinearFilter;
          const sm2 = new THREE.SpriteMaterial({ map: tex2, transparent: true, opacity: 0.5, depthTest: false });
          const sp2 = new THREE.Sprite(sm2);
          sp2.userData._origOpacity = 0.5;
          const sc2 = c2.width / c2.height;
          sp2.scale.set(sc2 * 3, 3, 1);
          sp2.position.y = -(r + 8);
          group.add(sp2);
        }

        _nodeThreeMap.set(node.id, group);
        return group;
      })
      .nodeLabel(() => '') // we render our own labels
      // --- Links: custom surface-following curves ---
      .linkThreeObject(link => {
        const group = new THREE.Group();
        const color = EDGE_COLORS[link.polarity] || '#8892a8';
        const colorInt = colorToInt(color);
        const w = link.weight || 0.5;

        // Curve line (will be updated in linkPositionUpdate)
        const NUM_CURVE_PTS = 24;
        const positions = new Float32Array(NUM_CURVE_PTS * 3);
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const isTension = link.polarity === 'tension';
        const lineMat = isTension
          ? new THREE.LineDashedMaterial({
              color: colorInt,
              transparent: true,
              opacity: 0.35 + w * 0.3,
              dashSize: 4,
              gapSize: 3,
            })
          : new THREE.LineBasicMaterial({
              color: colorInt,
              transparent: true,
              opacity: 0.35 + w * 0.3,
            });
        const lineOpacity = 0.35 + w * 0.3;
        const line = new THREE.Line(lineGeo, lineMat);
        line.userData._origOpacity = lineOpacity;
        group.add(line);

        // Arrow cone at 85% along the curve
        const arrowGeo = new THREE.ConeGeometry(1.2, 5, 8);
        arrowGeo.translate(0, 2.5, 0); // tip at origin after rotation
        arrowGeo.rotateX(Math.PI / 2);
        const arrowOpacity = 0.5 + w * 0.3;
        const arrowMat = new THREE.MeshBasicMaterial({
          color: colorInt,
          transparent: true,
          opacity: arrowOpacity,
        });
        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
        arrow.userData._origOpacity = arrowOpacity;
        group.add(arrow);

        group.userData = { numPoints: NUM_CURVE_PTS };
        _linkThreeMap.set(link.id, group);
        return group;
      })
      .linkPositionUpdate((groupObj, { start, end }, link) => {
        const N = groupObj.userData.numPoints;
        const line = groupObj.children[0];
        const arrow = groupObj.children[1];
        const posAttr = line.geometry.attributes.position;

        // Compute surface-following curve points
        for (let i = 0; i < N; i++) {
          const t = i / (N - 1);
          const pt = trumpetSurfacePoint(start, end, t);
          posAttr.setXYZ(i, pt.x, pt.y, pt.z);
        }
        posAttr.needsUpdate = true;
        // Recompute line distances for dashed materials
        if (line.material.isLineDashedMaterial) line.computeLineDistances();

        // Position arrow at 85% along the curve
        const arrowT = 0.85;
        const arrowPt = trumpetSurfacePoint(start, end, arrowT);
        const arrowPtAhead = trumpetSurfacePoint(start, end, Math.min(1, arrowT + 0.05));
        arrow.position.set(arrowPt.x, arrowPt.y, arrowPt.z);
        // Orient arrow along tangent
        arrow.lookAt(arrowPtAhead.x, arrowPtAhead.y, arrowPtAhead.z);

        return true; // we handled positioning
      })
      .linkDirectionalArrowLength(0) // disabled — we draw our own arrows
      // --- Forces ---
      .d3AlphaDecay(0.015)
      .d3VelocityDecay(0.45)
      // --- Interaction ---
      .onNodeClick(node => {
        selectedNode = graphData.nodes.find(n => n.id === node.id) || node;
        selectedEdge = null;
        renderDetailPanel();
        updateHighlight();
      })
      .onBackgroundClick(() => {
        selectedNode = null;
        selectedEdge = null;
        addEdgeMode = false;
        addEdgeSource = null;
        renderDetailPanel();
        updateHighlight();
      })
      .onLinkClick(link => {
        selectedEdge = graphData.edges.find(e => e.id === link.id) || link;
        selectedNode = null;
        renderDetailPanel();
        updateHighlight();
      })
      .onNodeHover(node => { hoveredNode = node; })
      .enableNodeDrag(true);

    const fg = graph3d;

    // --- Configure forces ---
    // Link forces operate tangentially on the surface; d3 sees Cartesian coords
    // but the tick handler projects everything back onto the trumpet each frame.
    fg.d3Force('link')
      .distance(link => {
        const w = link.weight || 0.5;
        const baseDist = (60 - w * 20) * trumpetScale;
        // Account for Z-separation so cross-layer links don't constantly pull
        // nodes together angularly (link can never reach target if Z-gap > target)
        const srcZ = (typeof link.source === 'object' ? link.source.z : 0) || 0;
        const tgtZ = (typeof link.target === 'object' ? link.target.z : 0) || 0;
        const zGap = Math.abs(srcZ - tgtZ);
        return Math.max(baseDist, zGap * 1.05);
      })
      .strength(link => {
        const w = link.weight || 0.5;
        return link.polarity === 'tension' ? 0.06 : 0.10 + w * 0.10;
      });

    // Circumference-aware charge: crowded layers (tip) get stronger repulsion
    // Pre-compute layer node counts for crowding factor
    const depthsForCharge = getComputedDepths();
    const layerBuckets = {};
    nodes3d.forEach(n => {
      const d = depthsForCharge.get(n.id) ?? 0.5;
      const bucket = Math.round(d * 10);
      layerBuckets[bucket] = (layerBuckets[bucket] || 0) + 1;
    });

    fg.d3Force('charge').strength(node => {
      const deg = degrees[node.id] || 0;
      const depth = depthsForCharge.get(node.id) ?? 0.5;
      const z = -diskRadius() * (1 - Math.exp(-2.5 * depth)) / (1 - Math.exp(-2.5));
      const circumference = 2 * Math.PI * trumpetRadius(z);
      const bucket = Math.round(depth * 10);
      const layerCount = layerBuckets[bucket] || 1;
      const nodeDiam = 2 * nodeRadius(node, degrees);
      const availablePerNode = circumference / layerCount;
      const crowding = Math.min(4, Math.max(1, (nodeDiam * 2.5) / availablePerNode));
      // Depth-based floor ensures bell nodes spread around full circumference;
      // crowding boost ensures tip nodes don't overlap
      const scale = Math.max(1 + depth * 0.8, crowding);
      return -(80 + deg * 15) * scale;
    }).distanceMax(bellRadius());

    // Remove default center + XY centering — surface projection handles radial positioning
    fg.d3Force('center', null);
    fg.d3Force('x', null);
    fg.d3Force('y', null);

    // Seed initial positions ON the trumpet surface — per-layer angular distribution
    // Group nodes by integer depth layer, distribute evenly within each layer,
    // offset layers by golden angle to prevent column alignment.
    {
      const depths = getComputedDepths();
      const layerMap = new Map(); // integerLayer → [node, ...]
      nodes3d.forEach(n => {
        const d = depths.get(n.id) ?? 0.5;
        const layerKey = Math.round(d * 10); // coarse buckets
        if (!layerMap.has(layerKey)) layerMap.set(layerKey, []);
        layerMap.get(layerKey).push(n);
      });
      const GOLDEN_ANGLE = 2.399963;
      let layerIndex = 0;
      for (const [, layerNodes] of layerMap) {
        const layerOffset = layerIndex * GOLDEN_ANGLE;
        layerNodes.forEach((n, i) => {
          n.z = nodeDepthZ(n);
          const r = trumpetRadius(n.z);
          const angle = (i / layerNodes.length) * 2 * Math.PI + layerOffset + (Math.random() - 0.5) * 0.15;
          n.x = r * Math.cos(angle);
          n.y = r * Math.sin(angle);
        });
        layerIndex++;
      }
    }

    // Engine tick: project every node ONTO the trumpet surface each frame,
    // then apply tangential repulsion that Cartesian charge can't provide.
    const tickDepths = getComputedDepths();
    fg.onEngineTick(() => {
      nodes3d.forEach(n => {
        // 1. Pull Z toward target depth (strong spring)
        const targetZ = nodeDepthZ(n);
        n.vz = (n.vz || 0) + (targetZ - (n.z || 0)) * 0.15;

        // 2. Project XY onto the trumpet surface at this Z
        const surfaceR = trumpetRadius(n.z || 0);
        const xyR = Math.sqrt((n.x || 0) ** 2 + (n.y || 0) ** 2);
        if (xyR > 0.001) {
          // Snap radial distance to surface radius
          const scale = surfaceR / xyR;
          n.x *= scale;
          n.y *= scale;
          // Remove radial component of velocity (keep tangential)
          const nx = (n.x / surfaceR) || 0;
          const ny = (n.y / surfaceR) || 0;
          const radialV = (n.vx || 0) * nx + (n.vy || 0) * ny;
          n.vx = (n.vx || 0) - radialV * nx;
          n.vy = (n.vy || 0) - radialV * ny;
        } else {
          // Node at origin — give it a nudge onto the surface
          const angle = Math.random() * 2 * Math.PI;
          n.x = surfaceR * Math.cos(angle);
          n.y = surfaceR * Math.sin(angle);
        }
      });

      // 3. Tangential repulsion: Cartesian charge becomes purely radial (useless
      //    after projection) once nodes are >120° apart. Pairwise angular repulsion
      //    ensures nodes spread around the full circumference.
      const layerGroups = {};
      nodes3d.forEach(n => {
        const d = tickDepths.get(n.id) ?? 0.5;
        const bucket = Math.round(d * 10);
        if (!layerGroups[bucket]) layerGroups[bucket] = [];
        layerGroups[bucket].push(n);
      });

      for (const group of Object.values(layerGroups)) {
        if (group.length < 2) continue;
        const idealAngle = (2 * Math.PI) / group.length;

        for (let i = 0; i < group.length; i++) {
          const a = group[i];
          const thetaA = Math.atan2(a.y, a.x);

          for (let j = i + 1; j < group.length; j++) {
            const b = group[j];
            const thetaB = Math.atan2(b.y, b.x);

            let dTheta = thetaA - thetaB;
            if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
            if (dTheta < -Math.PI) dTheta += 2 * Math.PI;

            const absDTheta = Math.abs(dTheta);
            if (absDTheta < 0.001) continue;

            // Only push when closer than 1.5x ideal spacing; once spread, stop
            if (absDTheta >= idealAngle * 1.5) continue;

            const push = 0.15 * (1 - absDTheta / (idealAngle * 1.5));
            const sign = dTheta > 0 ? 1 : -1;

            // Tangential direction at each node (perpendicular to radius)
            const tanAx = -Math.sin(thetaA);
            const tanAy = Math.cos(thetaA);
            a.vx = (a.vx || 0) + sign * push * tanAx;
            a.vy = (a.vy || 0) + sign * push * tanAy;

            const tanBx = -Math.sin(thetaB);
            const tanBy = Math.cos(thetaB);
            b.vx = (b.vx || 0) - sign * push * tanBx;
            b.vy = (b.vy || 0) - sign * push * tanBy;
          }
        }
      }
    });

    // Camera: look at the trumpet from an angle that shows the flare
    setTimeout(() => {
      if (graph3d) {
        graph3d.cameraPosition(
          { x: 200 * trumpetScale, y: -120 * trumpetScale, z: 180 * trumpetScale },   // camera position
          { x: 0, y: 0, z: -trumpetLen() * 0.4 }, // look-at point (middle of trumpet)
          2000  // animation duration ms
        );
      }
    }, 600);

    // Right-click drag = grab and slide the trumpet across the screen
    const controls = graph3d.controls();
    if (controls) {
      controls.screenSpacePanning = true;  // pan in screen plane, not ground plane
      controls.panSpeed = 0.5;             // near 1:1 feel without overshooting
      controls.rotateSpeed = 0.5;          // gentler left-click orbit
    }
  }

  // In 3D mode, drawGraph re-feeds data to the graph library
  // -----------------------------------------------------------------------
  // Selection highlight — distance-based dimming from selected node
  // -----------------------------------------------------------------------

  /**
   * BFS from `startId`, returning Map<nodeId, hopDistance>.
   * Follows all edges (support + tension) in both directions.
   * Unreachable nodes won't appear in the map.
   */
  function computeDistancesFrom(startId) {
    const dist = new Map();
    if (!graphData) return dist;
    // Build undirected adjacency
    const adj = {};
    (graphData.nodes || []).forEach(n => { adj[n.id] = []; });
    (graphData.edges || []).forEach(e => {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      if (adj[s]) adj[s].push(t);
      if (adj[t]) adj[t].push(s);
    });
    // BFS
    const queue = [startId];
    dist.set(startId, 0);
    let head = 0;
    while (head < queue.length) {
      const id = queue[head++];
      const d = dist.get(id);
      for (const neighbor of (adj[id] || [])) {
        if (!dist.has(neighbor)) {
          dist.set(neighbor, d + 1);
          queue.push(neighbor);
        }
      }
    }
    return dist;
  }

  /**
   * Map hop-distance to an opacity multiplier [0.08, 1.0].
   * d=0 (selected): 1.0, d=1: 1.0, d=2: 0.65, d=3: 0.35, d>=4: 0.12
   * Unreachable: 0.08
   */
  function distanceToOpacityFactor(d) {
    if (d === undefined || d === null) return 0.08;  // unreachable
    if (d <= 1) return 1.0;
    if (d === 2) return 0.65;
    if (d === 3) return 0.35;
    return 0.12;
  }

  /**
   * Update opacity on live Three.js objects to reflect selection state.
   * Distance-based: nearby nodes bright, far nodes dim, nothing selected = restore all.
   */
  function updateHighlight() {
    if (!graph3d) return;

    const distances = selectedNode ? computeDistancesFrom(selectedNode.id) : null;

    function setGroupOpacity(group, factor) {
      if (!group) return;
      group.traverse(child => {
        if (child.material) {
          const orig = child.userData._origOpacity ?? child.material.opacity;
          if (child.userData._origOpacity === undefined) {
            child.userData._origOpacity = child.material.opacity;
          }
          child.material.opacity = factor === null ? orig : orig * factor;
          child.material.needsUpdate = true;
        }
      });
    }

    // Update nodes
    (graphData.nodes || []).forEach(node => {
      const group = _nodeThreeMap.get(node.id);
      if (!distances) {
        // No selection — restore
        setGroupOpacity(group, null);
      } else {
        const d = distances.get(node.id);
        setGroupOpacity(group, distanceToOpacityFactor(d));
      }
    });

    // Update links — use the farther endpoint's distance
    (graphData.edges || []).forEach(edge => {
      const group = _linkThreeMap.get(edge.id);
      if (!distances) {
        setGroupOpacity(group, null);
      } else {
        const s = typeof edge.source === 'object' ? edge.source.id : edge.source;
        const t = typeof edge.target === 'object' ? edge.target.id : edge.target;
        const ds = distances.get(s);
        const dt = distances.get(t);
        // Link brightness = worst of the two endpoints
        const maxDist = (ds === undefined || dt === undefined)
          ? undefined
          : Math.max(ds, dt);
        setGroupOpacity(group, distanceToOpacityFactor(maxDist));
      }
    });
  }

  function drawGraph() {
    if (!graph3d || !graphData) return;
    invalidateDegreeCache();
    _nodeThreeMap.clear();
    _linkThreeMap.clear();
    const degrees = computeDegrees();
    // Recompute trumpet scale for live resizing as nodes are added
    trumpetScale = computeTrumpetScale(graphData.nodes.length);
    const nodes3d = graphData.nodes.map(n => ({
      ...n,
      _color: nodeColor(n),
      _radius: nodeRadius(n, degrees),
      _opacity: CONFIDENCE_OPACITY[n.confidence] || 0.7,
    }));
    const links3d = graphData.edges.map(e => ({
      ...e,
      source: typeof e.source === 'object' ? e.source.id : e.source,
      target: typeof e.target === 'object' ? e.target.id : e.target,
    }));
    graph3d.graphData({ nodes: nodes3d, links: links3d });
    // Reheat so nodes re-settle onto the (possibly resized) trumpet
    if (graph3d) graph3d.d3ReheatSimulation();
  }

  function stopSimulation() {
    // 3d-force-graph manages its own simulation
  }

  function cancelAnimFrame() {
    // No manual animation frames in 3D mode
  }

  function zoomToFit() {
    if (graph3d) {
      graph3d.zoomToFit(800, 60);
    }
  }

  function updateCanvasCursor() {
    // 3d-force-graph handles cursors
  }

  // -----------------------------------------------------------------------
  // Domain drill-down
  // -----------------------------------------------------------------------

  function enterDomainScope(id) {
    domainScope = id;
    selectedNode = null;
    selectedEdge = null;
    stopSimulation();
    cancelAnimFrame();
    renderGraph();
  }

  function exitDomainScope() {
    domainScope = null;
    selectedNode = null;
    selectedEdge = null;
    stopSimulation();
    cancelAnimFrame();
    renderGraph();
  }

  // -----------------------------------------------------------------------
  // Detail Panel
  // -----------------------------------------------------------------------

  function renderDetailPanel() {
    const container = document.getElementById('detail-panel-container');
    if (!container) return;
    if (!selectedNode && !selectedEdge) { container.innerHTML = ''; return; }
    if (selectedNode) renderNodeDetailPanel(container);
    else renderEdgeDetailPanel(container);
  }

  async function renderNodeDetailPanel(container) {
    const node = selectedNode;
    if (!node) return;

    const connections = [];
    graphData.edges.forEach(e => {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      if (s === node.id) {
        connections.push({ direction: 'out', polarity: e.polarity, node: graphData.nodes.find(n => n.id === t), edge: e });
      } else if (t === node.id) {
        connections.push({ direction: 'in', polarity: e.polarity, node: graphData.nodes.find(n => n.id === s), edge: e });
      }
    });

    let html = `<div class="detail-panel">`;
    html += `<div class="detail-panel-header">
      <h2>${escapeHtml(node.label)}</h2>
      <button class="detail-panel-close" id="detail-close">&times;</button>
    </div>`;

    html += `<div class="detail-panel-meta">
      ${confidenceBadge(node.confidence)}
      ${tierBadge(node.tier)}
    </div>`;

    html += `<div class="detail-panel-content" id="detail-content"><div class="loading">Loading...</div></div>`;

    if (connections.length > 0) {
      html += `<div class="detail-connections"><h3>Connections</h3>`;
      for (const c of connections) {
        const arrow = c.direction === 'out' ? '&rarr;' : '&larr;';
        // Edge convention: source grounds/supports target
        // out = this node supports the other; in = this node is supported by the other
        let relLabel;
        if (c.polarity === 'tension') {
          relLabel = 'tension';
        } else {
          relLabel = c.direction === 'out' ? 'supports' : 'supported';
        }
        html += `<div class="detail-connection-item" data-node-id="${c.node ? c.node.id : ''}">
          <span class="conn-arrow">${arrow}</span>
          <span class="conn-polarity polarity-${c.polarity}">${relLabel}</span>
          <span>${escapeHtml(c.node ? c.node.label : '?')}</span>
        </div>`;
      }
      html += `</div>`;
    }

    html += `<div class="detail-actions">
      <button class="btn btn-small" id="btn-examine">Examine</button>
      <button class="btn btn-small" id="btn-contest">Contest</button>
      <button class="btn btn-small" id="btn-add-edge-from">Add Edge</button>
      ${node.type === 'domain' ? `<button class="btn btn-small" id="btn-decompose">Decompose</button>` : ''}
    </div>`;

    html += `</div>`;
    container.innerHTML = html;

    // Load content
    if (node.contentPath) {
      try {
        const data = await fetchFile(node.contentPath);
        const el = document.getElementById('detail-content');
        if (el) el.innerHTML = `<div class="md-content">${renderMarkdown(data.content || '')}</div>`;
      } catch {
        const el = document.getElementById('detail-content');
        if (el) el.innerHTML = '<p style="color:var(--text-dim);font-size:0.82rem">No content yet.</p>';
      }
    } else {
      const el = document.getElementById('detail-content');
      if (el) el.innerHTML = '<p style="color:var(--text-dim);font-size:0.82rem">No content file.</p>';
    }

    // Events
    document.getElementById('detail-close')?.addEventListener('click', () => {
      selectedNode = null; selectedEdge = null; renderDetailPanel(); updateHighlight();
    });
    document.getElementById('btn-examine')?.addEventListener('click', () => {
      navigate(node.contentPath ? `dialogue?context=${encodeURIComponent(node.contentPath)}` : 'dialogue');
    });
    document.getElementById('btn-contest')?.addEventListener('click', () => {
      navigate(`dialogue?contest=${encodeURIComponent(node.id)}`);
    });
    document.getElementById('btn-add-edge-from')?.addEventListener('click', () => {
      addEdgeMode = true; addEdgeSource = node; updateCanvasCursor();
    });
    document.getElementById('btn-decompose')?.addEventListener('click', () => showDecomposeModal(node));

    container.querySelectorAll('.detail-connection-item').forEach(el => {
      el.addEventListener('click', () => {
        const id = el.getAttribute('data-node-id');
        const target = graphData.nodes.find(n => n.id === id);
        if (target) { selectedNode = target; selectedEdge = null; renderDetailPanel(); updateHighlight(); }
      });
    });
  }

  function renderEdgeDetailPanel(container) {
    const edge = selectedEdge;
    if (!edge) return;

    const s = typeof edge.source === 'object' ? edge.source.id : edge.source;
    const t = typeof edge.target === 'object' ? edge.target.id : edge.target;
    const srcNode = graphData.nodes.find(n => n.id === s);
    const tgtNode = graphData.nodes.find(n => n.id === t);
    const weight = edge.weight ?? 0.5;

    let html = `<div class="detail-panel">`;
    html += `<div class="detail-panel-header">
      <h2>${escapeHtml(srcNode ? srcNode.label : s)} &rarr; ${escapeHtml(tgtNode ? tgtNode.label : t)}</h2>
      <button class="detail-panel-close" id="detail-close">&times;</button>
    </div>`;

    html += `<div class="detail-panel-meta">
      <span class="conn-polarity polarity-${edge.polarity}">${edge.polarity}</span>
    </div>`;

    html += `<div class="edge-weight-display">
      <span style="font-size:11px;color:var(--text-dim)">Weight</span>
      <div class="edge-weight-bar">
        <div class="edge-weight-fill polarity-${edge.polarity}" style="width:${weight * 100}%"></div>
      </div>
      <span class="edge-weight-label">${weight.toFixed(2)}</span>
    </div>`;

    html += `<div class="modal-field">
      <label>Adjust Weight</label>
      <div class="weight-slider-container">
        <input type="range" id="edge-weight-slider" min="0" max="1" step="0.05" value="${weight}">
        <span class="weight-value" id="edge-weight-value">${weight.toFixed(2)}</span>
      </div>
    </div>`;

    if (edge.description) {
      html += `<div class="detail-panel-content">
        <p style="font-size:0.85rem;color:var(--text-dim);line-height:1.55">${escapeHtml(edge.description)}</p>
      </div>`;
    }

    if (edge.contentPath) {
      html += `<div class="detail-panel-content" id="edge-detail-content"><div class="loading">Loading...</div></div>`;
    }

    html += `<div class="detail-actions">
      <button class="btn btn-small btn-danger" id="btn-delete-edge">Delete Edge</button>
    </div></div>`;

    container.innerHTML = html;

    if (edge.contentPath) {
      fetchFile(edge.contentPath).then(data => {
        const el = document.getElementById('edge-detail-content');
        if (el) el.innerHTML = `<div class="md-content">${renderMarkdown(data.content || '')}</div>`;
      }).catch(() => {
        const el = document.getElementById('edge-detail-content');
        if (el) el.innerHTML = '';
      });
    }

    document.getElementById('detail-close')?.addEventListener('click', () => {
      selectedNode = null; selectedEdge = null; renderDetailPanel(); updateHighlight();
    });

    const slider = document.getElementById('edge-weight-slider');
    slider?.addEventListener('input', (e) => {
      document.getElementById('edge-weight-value').textContent = parseFloat(e.target.value).toFixed(2);
    });
    slider?.addEventListener('change', async (e) => {
      const val = parseFloat(e.target.value);
      edge.weight = val;
      const ge = graphData.edges.find(x => x.id === edge.id);
      if (ge) ge.weight = val;
      try { await patchEdge({ id: edge.id, weight: val }); } catch (err) { console.error(err); }
      renderEdgeDetailPanel(container);
      drawGraph();
    });

    document.getElementById('btn-delete-edge')?.addEventListener('click', () => handleDeleteEdge(edge.id));
  }

  async function handleDeleteEdge(edgeId) {
    if (!confirm('Delete this edge?')) return;
    try {
      await deleteEdgeApi(edgeId);
      graphData.edges = graphData.edges.filter(e => e.id !== edgeId);
      selectedEdge = null;
      stopSimulation();
      cancelAnimFrame();
      renderGraph();
    } catch (err) { alert('Failed: ' + err.message); }
  }

  // -----------------------------------------------------------------------
  // Add Node Modal
  // -----------------------------------------------------------------------

  function showAddNodeModal() {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';

    const domainOptions = graphData.nodes.filter(n => n.type === 'domain')
      .map(n => `<option value="${n.id}">${escapeHtml(n.label)}</option>`).join('');

    overlay.innerHTML = `
      <div class="modal">
        <h2>Add Belief</h2>
        <div class="modal-field"><label>Label</label><input type="text" id="new-node-label" placeholder="e.g., Moral Luck"></div>
        <div class="modal-field"><label>Type</label><select id="new-node-type"><option value="domain">Domain (decomposable)</option><option value="belief">Belief (leaf)</option></select></div>
        <div class="modal-field"><label>Tier</label><select id="new-node-tier"><option value="">Unassigned</option><option value="metaethics">Metaethics</option><option value="normative">Normative</option><option value="applied">Applied</option></select></div>
        <div class="modal-field"><label>Confidence</label><select id="new-node-confidence"><option value="tentative">Tentative</option><option value="probable">Probable</option><option value="certain">Certain</option><option value="under-revision">Under Revision</option></select></div>
        <div class="modal-field"><label>Parent Domain</label><select id="new-node-parent"><option value="">None (top-level)</option>${domainOptions}</select></div>
        <div class="modal-actions"><button class="btn" id="modal-cancel">Cancel</button><button class="btn btn-primary" id="modal-save">Add</button></div>
      </div>`;

    document.body.appendChild(overlay);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('#modal-cancel').addEventListener('click', () => overlay.remove());

    overlay.querySelector('#modal-save').addEventListener('click', async () => {
      const label = document.getElementById('new-node-label').value.trim();
      if (!label) { alert('Label required.'); return; }
      const id = slugify(label);
      const type = document.getElementById('new-node-type').value;
      const tier = document.getElementById('new-node-tier').value || null;
      const confidence = document.getElementById('new-node-confidence').value;
      const parent = document.getElementById('new-node-parent').value || null;
      const contentPath = `commitments/${id}.md`;
      const node = { id, label, type, tier, confidence, contentPath, parentDomain: parent, x: null, y: null };
      try {
        await patchNode(node);
        await saveFile(contentPath, `# ${label}\n\n**Confidence**: ${titleCase(confidence)}\n\n## Position\n\n*To be developed.*\n`);
        upsertNode(node);
        stopSimulation(); cancelAnimFrame();
        renderGraph();
        overlay.remove();
      } catch (err) { alert('Failed: ' + err.message); }
    });

    document.getElementById('new-node-label')?.focus();
  }

  // -----------------------------------------------------------------------
  // Add Edge Modal
  // -----------------------------------------------------------------------

  function showAddEdgeModal(srcNode, tgtNode) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `
      <div class="modal">
        <h2>Add Edge</h2>
        <p style="font-size:0.85rem;color:var(--text-dim);margin-bottom:0.85rem">
          <strong>${escapeHtml(srcNode.label)}</strong> &rarr; <strong>${escapeHtml(tgtNode.label)}</strong>
        </p>
        <div class="modal-field"><label>Polarity</label>
          <div class="polarity-toggle"><button id="pol-support" class="active-support">Support</button><button id="pol-tension">Tension</button></div>
        </div>
        <div class="modal-field"><label>Weight</label>
          <div class="weight-slider-container"><input type="range" id="new-edge-weight" min="0" max="1" step="0.05" value="0.5"><span class="weight-value" id="new-edge-weight-val">0.50</span></div>
        </div>
        <div class="modal-field"><label>Description</label><textarea id="new-edge-desc" placeholder="Describe the relationship..."></textarea></div>
        <div class="modal-actions"><button class="btn" id="modal-cancel">Cancel</button><button class="btn btn-primary" id="modal-save">Add</button></div>
      </div>`;

    document.body.appendChild(overlay);
    let polarity = 'support';

    const sBtn = overlay.querySelector('#pol-support');
    const tBtn = overlay.querySelector('#pol-tension');
    sBtn.addEventListener('click', () => { polarity = 'support'; sBtn.className = 'active-support'; tBtn.className = ''; });
    tBtn.addEventListener('click', () => { polarity = 'tension'; tBtn.className = 'active-tension'; sBtn.className = ''; });
    overlay.querySelector('#new-edge-weight').addEventListener('input', (e) => {
      overlay.querySelector('#new-edge-weight-val').textContent = parseFloat(e.target.value).toFixed(2);
    });
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('#modal-cancel').addEventListener('click', () => overlay.remove());
    overlay.querySelector('#modal-save').addEventListener('click', async () => {
      const weight = parseFloat(overlay.querySelector('#new-edge-weight').value);
      const description = overlay.querySelector('#new-edge-desc').value.trim();
      const id = `e-${srcNode.id}-${tgtNode.id}-${polarity}`;
      const edge = { id, source: srcNode.id, target: tgtNode.id, polarity, weight, description, contentPath: null };
      try {
        await patchEdge(edge);
        upsertEdge(edge);
        stopSimulation(); cancelAnimFrame();
        renderGraph();
        overlay.remove();
      } catch (err) { alert('Failed: ' + err.message); }
    });
  }

  // -----------------------------------------------------------------------
  // Decompose Modal
  // -----------------------------------------------------------------------

  function showDecomposeModal(parentNode) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `
      <div class="modal">
        <h2>Decompose: ${escapeHtml(parentNode.label)}</h2>
        <p style="font-size:0.85rem;color:var(--text-dim);margin-bottom:0.85rem">Add a sub-belief within this domain.</p>
        <div class="modal-field"><label>Sub-belief Label</label><input type="text" id="sub-label" placeholder="e.g., Rule vs. Act Consequentialism"></div>
        <div class="modal-field"><label>Confidence</label><select id="sub-confidence"><option value="tentative">Tentative</option><option value="probable">Probable</option><option value="certain">Certain</option><option value="under-revision">Under Revision</option></select></div>
        <div class="modal-actions"><button class="btn" id="modal-cancel">Cancel</button><button class="btn btn-primary" id="modal-save">Create</button></div>
      </div>`;

    document.body.appendChild(overlay);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('#modal-cancel').addEventListener('click', () => overlay.remove());
    overlay.querySelector('#modal-save').addEventListener('click', async () => {
      const label = overlay.querySelector('#sub-label').value.trim();
      if (!label) { alert('Label required.'); return; }
      const id = slugify(label);
      const confidence = overlay.querySelector('#sub-confidence').value;
      const contentPath = `commitments/${id}.md`;
      const node = { id, label, type: 'belief', tier: parentNode.tier, confidence, contentPath, parentDomain: parentNode.id, x: null, y: null };
      const edge = { id: `e-${parentNode.id}-${id}-support`, source: parentNode.id, target: id, polarity: 'support', weight: 0.5, description: `${label} is a sub-belief of ${parentNode.label}.`, contentPath: null };
      try {
        await patchNode(node);
        await patchEdge(edge);
        await saveFile(contentPath, `# ${label}\n\n**Confidence**: ${titleCase(confidence)}\n\n## Position\n\n*To be developed through dialogue.*\n`);
        upsertNode(node);
        upsertEdge(edge);
        stopSimulation(); cancelAnimFrame();
        renderGraph();
        overlay.remove();
      } catch (err) { alert('Failed: ' + err.message); }
    });
    overlay.querySelector('#sub-label')?.focus();
  }

  // -----------------------------------------------------------------------
  // Dialogue
  // -----------------------------------------------------------------------

  async function renderDialogue() {
    const hash = window.location.hash.slice(1);
    const qi = hash.indexOf('?');
    let contextPath = null;
    let contestId = null;
    if (qi !== -1) {
      const params = new URLSearchParams(hash.slice(qi + 1));
      contextPath = params.get('context');
      contestId = params.get('contest');
    }

    // Contest mode setup
    if (contestId) {
      if (!contestNode || contestNode.id !== contestId) {
        // Load graph if not available
        if (!graphData) {
          try { graphData = await fetchGraph(); } catch { /* fall through */ }
        }
        if (graphData) {
          const node = graphData.nodes.find(n => n.id === contestId);
          if (node) {
            contestNode = node;
            contestMode = true;
            contestSubgraph = gatherContestSubgraph(node.id);
            dialogueHistory = [];
            dialogueContext = null;
            dialogueContextContent = null;
          }
        }
      }
    } else if (contestMode) {
      // Navigated away from contest
      contestNode = null;
      contestSubgraph = null;
      contestMode = false;
    }

    if (!contestMode) {
      if (contextPath !== dialogueContext) {
        dialogueContext = contextPath;
        dialogueContextContent = null;
        dialogueHistory = [];
      }

      if (contextPath && !dialogueContextContent) {
        try { const d = await fetchFile(contextPath); dialogueContextContent = d.content || ''; }
        catch { dialogueContextContent = null; }
      }
    }

    renderDialogueUI();
  }

  function gatherContestSubgraph(nodeId) {
    if (!graphData) return null;
    const edges = graphData.edges;
    const nodesById = {};
    graphData.nodes.forEach(n => { nodesById[n.id] = n; });

    // 1st order
    const firstOrderEdges = [];
    const firstOrderNodeIds = new Set();
    for (const e of edges) {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      if (s === nodeId || t === nodeId) {
        firstOrderEdges.push(e);
        firstOrderNodeIds.add(s);
        firstOrderNodeIds.add(t);
      }
    }
    firstOrderNodeIds.delete(nodeId);

    // 2nd order
    const firstOrderEdgeIds = new Set(firstOrderEdges.map(e => e.id));
    const secondOrderEdges = [];
    const secondOrderNodeIds = new Set();
    for (const e of edges) {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      if (firstOrderEdgeIds.has(e.id)) continue;
      if (firstOrderNodeIds.has(s) || firstOrderNodeIds.has(t)) {
        secondOrderEdges.push(e);
        secondOrderNodeIds.add(s);
        secondOrderNodeIds.add(t);
      }
    }
    // Remove overlap
    firstOrderNodeIds.forEach(id => secondOrderNodeIds.delete(id));
    secondOrderNodeIds.delete(nodeId);

    return {
      nodes: [...firstOrderNodeIds, ...secondOrderNodeIds].map(id => nodesById[id]).filter(Boolean),
      firstOrderEdges,
      secondOrderEdges,
      firstOrderNodeIds,
      secondOrderNodeIds,
    };
  }

  async function renderDialogueUI() {
    await ensureActivePersona();
    const personas = await fetchPersonas().catch(() => []);
    const personaName = activePersonaData ? activePersonaData.name : 'Philosopher';

    let html = `<div class="dialogue-container">`;

    // Persona selector bar
    html += `<div class="persona-selector-bar">
      <label for="persona-select">Interlocutor:</label>
      <select id="persona-select">`;
    for (const p of personas) {
      const selected = p.id === activePersonaId ? ' selected' : '';
      html += `<option value="${escapeHtml(p.id)}"${selected}>${escapeHtml(p.name)}${p.builtin ? ' (built-in)' : ''}</option>`;
    }
    html += `</select>`;
    if (activePersonaData && !activePersonaData.builtin) {
      html += `<button class="btn btn-small btn-danger" id="delete-persona-btn" title="Delete this persona">Delete</button>`;
    }
    if (activePersonaData) {
      html += `<span class="persona-summary">${escapeHtml(activePersonaData.summary || '')}</span>`;
    }
    html += `</div>`;

    // Persona detail panel (collapsible)
    if (activePersonaData && (activePersonaData.tradition || (activePersonaData.key_positions && activePersonaData.key_positions.length > 0))) {
      const arrow = personaDetailExpanded ? '\u25BE' : '\u25B8';
      const bodyDisplay = personaDetailExpanded ? '' : ' style="display:none"';
      html += `<div class="persona-detail">
        <button class="persona-detail-toggle" id="persona-detail-toggle">
          ${activePersonaData.tradition ? `<span class="tradition-badge">${escapeHtml(activePersonaData.tradition)}</span>` : ''}
          <span class="persona-detail-summary">${escapeHtml(activePersonaData.summary || activePersonaData.name)}</span>
          <span class="toggle-arrow">${arrow}</span>
        </button>
        <div class="persona-detail-body" id="persona-detail-body"${bodyDisplay}>
          ${activePersonaData.key_positions && activePersonaData.key_positions.length > 0 ? `
            <h4>Key Positions</h4>
            <ul>${activePersonaData.key_positions.map(p => `<li>${escapeHtml(p)}</li>`).join('')}</ul>
          ` : ''}
        </div>
      </div>`;
    }

    if (contestMode && contestNode) {
      html += `<div class="contest-context-bar">
        <div class="contest-context-header">
          <span>Contesting: <strong>${escapeHtml(contestNode.label)}</strong></span>
          ${confidenceBadge(contestNode.confidence)}
          <button class="btn btn-small" onclick="window.app.clearContest()">Exit Contest</button>
        </div>`;
      if (contestSubgraph && contestSubgraph.nodes.length > 0) {
        html += `<div class="contest-connections-bar">`;
        for (const n of contestSubgraph.nodes) {
          const isFirst = contestSubgraph.firstOrderNodeIds.has(n.id);
          html += `<span class="contest-conn-chip ${isFirst ? 'first-order' : 'second-order'}">${escapeHtml(n.label)}</span>`;
        }
        html += `</div>`;
      }
      html += `</div>`;
    } else if (dialogueContext) {
      html += `<div class="dialogue-context-bar">
        <span>Examining: <span class="context-name">${escapeHtml(titleCase(fileBasename(dialogueContext)))}</span></span>
        <button class="btn btn-small" onclick="window.app.clearContext()">Clear</button>
      </div>`;
    }

    html += `<div class="dialogue-messages" id="dialogue-messages">`;
    if (dialogueHistory.length === 0) {
      html += `<div class="empty-state"><h3>Begin a dialogue</h3><p>Converse with ${escapeHtml(personaName)}. Pose questions, present commitments, or raise tensions.</p></div>`;
    } else {
      for (let i = 0; i < dialogueHistory.length; i++) {
        html += renderMessage(dialogueHistory[i], i === dialogueHistory.length - 1 && isStreaming && dialogueHistory[i].role === 'philosopher');
      }
    }
    html += `</div>`;

    html += `<div class="dialogue-input-area">
      <div class="dialogue-input-row">
        <textarea class="dialogue-textarea" id="dialogue-input" placeholder="Pose a question or present a position..." ${isStreaming ? 'disabled' : ''} rows="2"></textarea>
        <button class="btn btn-primary" id="dialogue-send" ${isStreaming ? 'disabled' : ''}>${isStreaming ? 'Thinking...' : 'Send'}</button>
      </div>
      <div class="dialogue-actions">
        ${dialogueHistory.length > 0 ? `
          ${contestMode ? `<button class="btn btn-small btn-primary" id="dialogue-finalize" ${isStreaming ? 'disabled' : ''}>Finalize Revision</button>` : ''}
          ${!contestMode ? `<button class="btn btn-small" id="dialogue-capture">Capture Insight</button>` : ''}
          <button class="btn btn-small" id="dialogue-save" ${isStreaming ? 'disabled' : ''}>Save</button>
          <button class="btn btn-small btn-danger" id="dialogue-clear" ${isStreaming ? 'disabled' : ''}>Clear</button>
        ` : ''}
      </div>
    </div>`;

    html += `<div id="capture-insight-container"></div></div>`;
    setContent(html);

    const msgs = document.getElementById('dialogue-messages');

    const input = document.getElementById('dialogue-input');
    const sendBtn = document.getElementById('dialogue-send');
    if (sendBtn && input) {
      sendBtn.addEventListener('click', () => handleSend(input));
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(input); } });
      input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 160) + 'px'; });
      if (!isStreaming) input.focus();
    }

    document.getElementById('dialogue-save')?.addEventListener('click', handleSaveDialogue);
    document.getElementById('dialogue-clear')?.addEventListener('click', () => {
      if (confirm('Clear? Unsaved messages will be lost.')) { dialogueHistory = []; renderDialogueUI(); }
    });
    document.getElementById('dialogue-capture')?.addEventListener('click', showCaptureInsightForm);
    document.getElementById('dialogue-finalize')?.addEventListener('click', handleFinalizeContest);

    // Persona selector
    document.getElementById('persona-select')?.addEventListener('change', async (e) => {
      const newId = e.target.value;
      if (newId === activePersonaId) return;
      if (dialogueHistory.length > 0 && !confirm('Switching persona will clear the current dialogue. Continue?')) {
        e.target.value = activePersonaId;
        return;
      }
      activePersonaId = newId;
      activePersonaData = null;
      dialogueHistory = [];
      renderDialogueUI();
    });
    document.getElementById('persona-detail-toggle')?.addEventListener('click', () => {
      personaDetailExpanded = !personaDetailExpanded;
      const body = document.getElementById('persona-detail-body');
      const toggle = document.getElementById('persona-detail-toggle');
      if (body) body.style.display = personaDetailExpanded ? '' : 'none';
      if (toggle) {
        const arrow = toggle.querySelector('.toggle-arrow');
        if (arrow) arrow.textContent = personaDetailExpanded ? '\u25BE' : '\u25B8';
      }
    });
    document.getElementById('delete-persona-btn')?.addEventListener('click', async () => {
      if (!confirm(`Delete persona "${activePersonaData.name}"? This cannot be undone.`)) return;
      try {
        await deletePersona(activePersonaId);
        activePersonaId = 'korsgaard';
        activePersonaData = null;
        dialogueHistory = [];
        renderDialogueUI();
      } catch (err) { alert('Failed to delete: ' + err.message); }
    });
  }

  function renderMessage(msg, streaming) {
    const isUser = msg.role === 'user' || msg.role === 'human';
    const cls = isUser ? 'message-user' : 'message-philosopher';
    const label = isUser ? 'You' : escapeHtml(activePersonaData ? activePersonaData.name : 'Philosopher');
    const content = `<div class="md-content">${renderMarkdown(msg.content || '')}</div>`;
    return `<div class="message ${cls} ${streaming ? 'message-streaming' : ''}">
      <div class="message-label">${label}</div>
      <div class="message-bubble">${content}</div>
    </div>`;
  }

  async function handleSend(input) {
    const message = input.value.trim();
    if (!message || isStreaming) return;

    dialogueHistory.push({ role: 'user', content: message });
    input.value = '';
    dialogueHistory.push({ role: 'philosopher', content: '' });
    isStreaming = true;
    renderDialogueUI();

    await ensureActivePersona();
    if (!activePersonaData) {
      dialogueHistory.pop();
      dialogueHistory.pop();
      isStreaming = false;
      alert('Could not load persona data. Please check your connection and try again.');
      renderDialogueUI();
      return;
    }
    let systemPrompt;
    if (contestMode && contestNode) {
      systemPrompt = activePersonaData.contest_prompt;
      // Add graph context for the contested belief
      let graphContext = `\n\n---\n\nBelief under contestation: "${contestNode.label}" (confidence: ${contestNode.confidence || 'unset'})`;
      if (contestSubgraph && contestSubgraph.nodes.length > 0) {
        graphContext += '\n\nConnected beliefs in their normative framework:';
        for (const n of contestSubgraph.nodes) {
          const isFirst = contestSubgraph.firstOrderNodeIds.has(n.id);
          graphContext += `\n- ${n.label} (${isFirst ? 'directly connected' : 'indirectly connected'}, confidence: ${n.confidence || 'unset'})`;
        }
      }
      systemPrompt += graphContext;
    } else {
      systemPrompt = activePersonaData.system_prompt;
      if (dialogueContextContent) systemPrompt += `\n\n---\n\nDocument under examination:\n\n${dialogueContextContent}`;
    }

    const historyForApi = dialogueHistory.slice(0, -2).map(m => ({
      role: m.role === 'user' ? 'human' : 'assistant',
      content: m.content,
    }));

    // Reset typewriter state
    typewriterQueue = '';
    typewriterRevealed = '';
    startTypewriter();

    try {
      const response = await sendDialogueRequest(systemPrompt, historyForApi, message);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();
        for (const event of events) {
          for (const line of event.split('\n')) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') break;
              fullText += data + '\n';
            }
          }
        }
        // Feed new text into the typewriter queue
        typewriterQueue = fullText;
        dialogueHistory[dialogueHistory.length - 1].content = fullText;
      }

      if (buffer.trim()) {
        for (const line of buffer.split('\n')) {
          if (line.startsWith('data: ')) {
            const d = line.slice(6);
            if (d !== '[DONE]') fullText += d;
          }
        }
        typewriterQueue = fullText;
        dialogueHistory[dialogueHistory.length - 1].content = fullText;
      }
    } catch (err) {
      dialogueHistory[dialogueHistory.length - 1].content = '*Error reaching the philosopher.*\n\n`' + err.message + '`';
      stopTypewriter();
      isStreaming = false;
      renderDialogueUI();
      return;
    }

    // SSE done — let typewriter finish draining the remaining characters
    await waitForTypewriterDone();
    isStreaming = false;
    renderDialogueUI();

  }

  function startTypewriter() {
    if (typewriterRafId) return;
    function tick() {
      if (typewriterRevealed.length < typewriterQueue.length) {
        // Reveal next batch of characters
        const end = Math.min(typewriterRevealed.length + CHARS_PER_FRAME, typewriterQueue.length);
        typewriterRevealed = typewriterQueue.slice(0, end);
        updateStreamingBubble(typewriterRevealed);
      }
      typewriterRafId = requestAnimationFrame(tick);
    }
    typewriterRafId = requestAnimationFrame(tick);
  }

  function stopTypewriter() {
    if (typewriterRafId) {
      cancelAnimationFrame(typewriterRafId);
      typewriterRafId = null;
    }
  }

  function waitForTypewriterDone() {
    return new Promise(resolve => {
      function check() {
        if (typewriterRevealed.length >= typewriterQueue.length) {
          stopTypewriter();
          resolve();
        } else {
          requestAnimationFrame(check);
        }
      }
      check();
    });
  }

  function updateStreamingBubble(text) {
    const msgs = document.getElementById('dialogue-messages');
    if (!msgs) return;
    const els = msgs.querySelectorAll('.message-philosopher');
    const last = els[els.length - 1];
    if (!last) return;
    const bubble = last.querySelector('.message-bubble');
    if (bubble) {
      bubble.innerHTML = `<div class="md-content">${renderMarkdown(text)}</div>`;
    }
    // Scrolling left to user
  }

  // -----------------------------------------------------------------------
  // Contest — Finalize, Diff Modal, Apply
  // -----------------------------------------------------------------------

  async function handleFinalizeContest() {
    if (!contestMode || !contestNode) return;
    if (dialogueHistory.length < 2) {
      alert('Have a conversation about the revision first.');
      return;
    }

    const btn = document.getElementById('dialogue-finalize');
    if (btn) { btn.disabled = true; btn.textContent = 'Evaluating...'; }

    try {
      const r = await fetch(`${API_BASE}/contest-node`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node_id: contestNode.id,
          history: dialogueHistory,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `Request failed: ${r.status}`);
      }
      const diff = await r.json();
      showRevisionDiffModal(diff);
    } catch (err) {
      alert('Evaluation failed: ' + err.message);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = 'Finalize Revision'; }
    }
  }

  function showRevisionDiffModal(diff) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';

    let html = `<div class="modal modal-wide"><h2>Proposed Revision</h2>`;

    if (diff.summary) {
      html += `<div class="revision-summary"><p>${escapeHtml(diff.summary)}</p></div>`;
    }

    const hasChanges = (diff.node_updates?.length || 0) + (diff.edge_updates?.length || 0) +
      (diff.edge_deletions?.length || 0) + (diff.new_nodes?.length || 0) + (diff.new_edges?.length || 0) > 0;

    if (!hasChanges) {
      html += `<p style="color:var(--text-dim);margin:1rem 0">No graph changes proposed.</p>`;
    } else {
      // Node updates
      if (diff.node_updates?.length > 0) {
        html += `<h3>Belief Updates</h3>`;
        for (const upd of diff.node_updates) {
          const node = graphData?.nodes.find(n => n.id === upd.id);
          const label = node ? node.label : upd.id;
          html += `<div class="revision-item revision-change">
            <strong>${escapeHtml(label)}</strong>
            ${upd.confidence ? ` &rarr; confidence: <em>${escapeHtml(upd.confidence)}</em>` : ''}
            ${upd.label && upd.label !== label ? ` &rarr; label: <em>${escapeHtml(upd.label)}</em>` : ''}
          </div>`;
        }
      }

      // Edge updates
      if (diff.edge_updates?.length > 0) {
        html += `<h3>Connection Changes</h3>`;
        for (const upd of diff.edge_updates) {
          html += `<div class="revision-item revision-change">
            <strong>${escapeHtml(upd.id)}</strong> &rarr; weight: ${upd.weight ?? '?'}
            ${upd.description ? `<div class="revision-reason">${escapeHtml(upd.description)}</div>` : ''}
          </div>`;
        }
      }

      // Edge deletions
      if (diff.edge_deletions?.length > 0) {
        html += `<h3>Connections Removed</h3>`;
        for (const id of diff.edge_deletions) {
          html += `<div class="revision-item revision-deletion">${escapeHtml(id)}</div>`;
        }
      }

      // New nodes
      if (diff.new_nodes?.length > 0) {
        html += `<h3>New Beliefs</h3>`;
        for (const n of diff.new_nodes) {
          html += `<div class="revision-item revision-addition">
            <strong>${escapeHtml(n.label || n.id)}</strong>
            (${escapeHtml(n.tier || '?')}, ${escapeHtml(n.confidence || '?')})
          </div>`;
        }
      }

      // New edges
      if (diff.new_edges?.length > 0) {
        html += `<h3>New Connections</h3>`;
        for (const e of diff.new_edges) {
          html += `<div class="revision-item revision-addition">
            ${escapeHtml(e.source)} &rarr; ${escapeHtml(e.target)} (${escapeHtml(e.polarity)}, weight: ${e.weight ?? 0.5})
            ${e.description ? `<div class="revision-reason">${escapeHtml(e.description)}</div>` : ''}
          </div>`;
        }
      }
    }

    html += `<div class="modal-actions">
      <button class="btn" id="revision-reject">Reject</button>
      ${hasChanges ? `<button class="btn btn-primary" id="revision-accept">Apply Revision</button>` : ''}
    </div></div>`;

    overlay.innerHTML = html;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('#revision-reject')?.addEventListener('click', () => overlay.remove());
    overlay.querySelector('#revision-accept')?.addEventListener('click', async () => {
      try {
        await applyRevisionDiff(diff);
        overlay.remove();
        // Clear contest state and navigate to graph
        contestNode = null;
        contestSubgraph = null;
        contestMode = false;
        dialogueHistory = [];
        navigate('graph');
      } catch (err) {
        alert('Failed to apply: ' + err.message);
      }
    });
  }

  async function applyRevisionDiff(diff) {
    // Apply node updates
    for (const upd of (diff.node_updates || [])) {
      const patch = { id: upd.id };
      if (upd.confidence) patch.confidence = upd.confidence;
      if (upd.label) patch.label = upd.label;
      await patchNode(patch);
      if (graphData) {
        const n = graphData.nodes.find(x => x.id === upd.id);
        if (n) {
          if (upd.confidence) n.confidence = upd.confidence;
          if (upd.label) n.label = upd.label;
        }
      }
    }

    // Apply edge updates
    for (const upd of (diff.edge_updates || [])) {
      const patch = { id: upd.id };
      if (upd.weight != null) patch.weight = upd.weight;
      if (upd.description) patch.description = upd.description;
      await patchEdge(patch);
      if (graphData) {
        const e = graphData.edges.find(x => x.id === upd.id);
        if (e) {
          if (upd.weight != null) e.weight = upd.weight;
          if (upd.description) e.description = upd.description;
        }
      }
    }

    // Apply edge deletions
    for (const id of (diff.edge_deletions || [])) {
      await deleteEdgeApi(id);
      if (graphData) {
        graphData.edges = graphData.edges.filter(e => e.id !== id);
      }
    }

    // Apply new nodes
    for (const n of (diff.new_nodes || [])) {
      const node = { ...n, x: null, y: null };
      if (!node.contentPath) node.contentPath = null;
      await patchNode(node);
      if (graphData) upsertNode(node);
    }

    // Apply new edges
    for (const e of (diff.new_edges || [])) {
      const edge = {
        id: e.id || `e-${e.source}-${e.target}-${e.polarity || 'support'}`,
        source: e.source,
        target: e.target,
        polarity: e.polarity || 'support',
        weight: e.weight ?? 0.5,
        description: e.description || '',
        contentPath: null,
      };
      await patchEdge(edge);
      if (graphData) upsertEdge(edge);
    }
  }

  async function handleSaveDialogue() {
    if (dialogueHistory.length === 0) return;
    const topic = prompt('Name this dialogue:');
    if (!topic) return;
    try { await saveDialogue(topic.trim(), dialogueHistory); alert('Saved.'); invalidateCache(); }
    catch (err) { alert('Failed: ' + err.message); }
  }

  // -----------------------------------------------------------------------
  // Capture Insight
  // -----------------------------------------------------------------------

  async function showCaptureInsightForm() {
    const container = document.getElementById('capture-insight-container');
    if (!container) return;

    if (!graphData) {
      try { graphData = await fetchGraph(); } catch { alert('Could not load graph.'); return; }
    }

    container.innerHTML = `
      <div class="capture-insight-bar">
        <h4>Capture Insight</h4>
        <div class="insight-tabs">
          <button class="insight-tab active" data-tab="ai">AI-Assisted</button>
          <button class="insight-tab" data-tab="manual">Manual</button>
        </div>
        <div id="insight-tab-content"></div>
        <div class="modal-actions">
          <button class="btn btn-small" id="insight-cancel">Cancel</button>
        </div>
      </div>`;

    const tabContent = document.getElementById('insight-tab-content');
    const tabs = container.querySelectorAll('.insight-tab');

    function switchTab(tabName) {
      tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
      if (tabName === 'ai') {
        renderAITab(tabContent);
      } else {
        renderManualInsightFields(tabContent);
      }
    }

    tabs.forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));
    document.getElementById('insight-cancel').addEventListener('click', () => { container.innerHTML = ''; });

    // Default: AI-Assisted tab
    switchTab('ai');
  }

  async function renderAITab(container) {
    container.innerHTML = `<div class="insight-loading"><div class="loading-spinner"></div>Analyzing recent exchanges\u2026</div>`;

    try {
      const resp = await fetch(`${API_BASE}/extract-candidates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: dialogueHistory, turn_window: 3 }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        container.innerHTML = `<div class="insight-error">${escapeHtml(err.error || 'Extraction failed')}</div>`;
        return;
      }

      const data = await resp.json();
      const candidates = data.candidates || [];

      if (candidates.length === 0) {
        container.innerHTML = `<div class="insight-empty">${escapeHtml(data.note || 'No insights detected in recent exchanges.')}</div>`;
        return;
      }

      renderCandidateList(container, candidates);
    } catch (err) {
      container.innerHTML = `<div class="insight-error">Network error: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderCandidateList(container, candidates) {
    let html = '<div class="candidate-list">';
    candidates.forEach((c, i) => {
      const checked = 'checked';
      const label = candidateLabel(c);
      const tier = c.tier || '';
      const confidence = c.new_confidence || c.confidence || '';
      const rationale = c.rationale || '';
      html += `
        <label class="candidate-item checked" data-idx="${i}">
          <input type="checkbox" class="candidate-check" data-idx="${i}" ${checked}>
          <div>
            <div class="candidate-label">${escapeHtml(label)}</div>
            <div class="candidate-meta">
              ${tier ? `<span class="candidate-tier ${escapeHtml(tier)}">${escapeHtml(tier)}</span>` : ''}
              ${confidence ? `<span>${escapeHtml(confidence)}</span>` : ''}
              <span>${escapeHtml(c.type)}</span>
            </div>
            ${rationale ? `<div class="candidate-rationale">${escapeHtml(rationale)}</div>` : ''}
          </div>
        </label>`;
    });
    html += '</div>';
    html += '<button class="btn btn-small btn-primary" id="insight-add-selected">Add Selected</button>';
    container.innerHTML = html;

    // Toggle checked styling
    container.querySelectorAll('.candidate-check').forEach(cb => {
      cb.addEventListener('change', () => {
        cb.closest('.candidate-item').classList.toggle('checked', cb.checked);
      });
    });

    document.getElementById('insight-add-selected').addEventListener('click', () => {
      handleAddSelectedCandidates(container, candidates);
    });
  }

  function candidateLabel(c) {
    if (c.type === 'new_node') return c.label || c.id || 'New belief';
    if (c.type === 'confidence_update') {
      const node = graphData?.nodes.find(n => n.id === c.node_id);
      return `Update: ${node ? node.label : c.node_id} \u2192 ${c.new_confidence}`;
    }
    if (c.type === 'tension') {
      const names = (c.between || []).map(id => {
        const node = graphData?.nodes.find(n => n.id === id);
        return node ? node.label : id;
      });
      return `Tension: ${names.join(' \u2194 ')}`;
    }
    return c.label || c.description || 'Insight';
  }

  async function handleAddSelectedCandidates(container, candidates) {
    const checks = container.querySelectorAll('.candidate-check');
    const selected = [];
    checks.forEach(cb => { if (cb.checked) selected.push(candidates[parseInt(cb.dataset.idx)]); });

    if (selected.length === 0) { alert('No candidates selected.'); return; }

    let added = 0;
    try {
      for (const c of selected) {
        if (c.type === 'new_node') {
          const id = c.id || slugify(c.label);
          const node = { id, label: c.label, type: 'domain', tier: c.tier || null, confidence: c.confidence || 'tentative', contentPath: `commitments/${id}.md`, parentDomain: null, x: null, y: null };
          await patchNode(node);
          await saveFile(node.contentPath, `# ${c.label}\n\n**Confidence**: ${titleCase(c.confidence || 'tentative')}\n\n## Position\n\n*Captured from dialogue.*\n`);
          upsertNode(node);
          // Add suggested edges
          if (c.suggested_edges && Array.isArray(c.suggested_edges)) {
            for (const se of c.suggested_edges) {
              const edgeId = `e-${se.source}-${se.target}-${se.polarity || 'support'}`;
              const edge = { id: edgeId, source: se.source, target: se.target, polarity: se.polarity || 'support', weight: se.weight || 0.5, description: '', contentPath: null };
              await patchEdge(edge);
              upsertEdge(edge);
            }
          }
          added++;
        } else if (c.type === 'confidence_update') {
          await patchNode({ id: c.node_id, confidence: c.new_confidence });
          const n = graphData?.nodes.find(x => x.id === c.node_id);
          if (n) n.confidence = c.new_confidence;
          added++;
        } else if (c.type === 'tension') {
          const src = c.between[0];
          const tgt = c.between[1];
          const edgeId = `e-${src}-${tgt}-tension`;
          const edge = { id: edgeId, source: src, target: tgt, polarity: 'tension', weight: 0.5, description: c.description || '', contentPath: null };
          await patchEdge(edge);
          upsertEdge(edge);
          added++;
        }
      }

      const insightContainer = document.getElementById('capture-insight-container');
      if (insightContainer) insightContainer.innerHTML = '';
      alert(`${added} insight${added !== 1 ? 's' : ''} captured.`);
      invalidateDegreeCache();
    } catch (err) {
      alert('Failed to add some candidates: ' + err.message);
    }
  }

  function renderManualInsightFields(container) {
    const opts = graphData.nodes.map(n => `<option value="${n.id}">${escapeHtml(n.label)}</option>`).join('');

    container.innerHTML = `
      <div class="modal-field"><label>Type</label>
        <select id="insight-type">
          <option value="support">A grounds B (support)</option>
          <option value="tension">A tensions B (tension)</option>
          <option value="new-belief">New belief</option>
          <option value="confidence">Confidence changed</option>
        </select>
      </div>
      <div id="insight-fields"></div>
      <button class="btn btn-small btn-primary" id="insight-save">Capture</button>`;

    function renderFields() {
      const type = document.getElementById('insight-type').value;
      const fields = document.getElementById('insight-fields');
      if (type === 'support' || type === 'tension') {
        fields.innerHTML = `
          <div class="modal-field"><label>Source</label><select id="insight-source">${opts}</select></div>
          <div class="modal-field"><label>Target</label><select id="insight-target">${opts}</select></div>
          <div class="modal-field"><label>Weight</label><div class="weight-slider-container"><input type="range" id="insight-weight" min="0" max="1" step="0.05" value="0.5"><span class="weight-value" id="insight-weight-val">0.50</span></div></div>
          <div class="modal-field"><label>Description</label><textarea id="insight-desc" placeholder="Describe..."></textarea></div>`;
        fields.querySelector('#insight-weight')?.addEventListener('input', (e) => {
          fields.querySelector('#insight-weight-val').textContent = parseFloat(e.target.value).toFixed(2);
        });
      } else if (type === 'new-belief') {
        fields.innerHTML = `
          <div class="modal-field"><label>Label</label><input type="text" id="insight-label" placeholder="e.g., Moral Luck"></div>
          <div class="modal-field"><label>Tier</label><select id="insight-tier"><option value="">Unassigned</option><option value="metaethics">Metaethics</option><option value="normative">Normative</option><option value="applied">Applied</option></select></div>
          <div class="modal-field"><label>Confidence</label><select id="insight-confidence"><option value="tentative">Tentative</option><option value="probable">Probable</option><option value="certain">Certain</option><option value="under-revision">Under Revision</option></select></div>`;
      } else if (type === 'confidence') {
        fields.innerHTML = `
          <div class="modal-field"><label>Node</label><select id="insight-node">${opts}</select></div>
          <div class="modal-field"><label>New Confidence</label><select id="insight-confidence"><option value="tentative">Tentative</option><option value="probable">Probable</option><option value="certain">Certain</option><option value="under-revision">Under Revision</option></select></div>`;
      }
    }

    document.getElementById('insight-type').addEventListener('change', renderFields);
    renderFields();

    document.getElementById('insight-save').addEventListener('click', async () => {
      const type = document.getElementById('insight-type').value;
      try {
        if (type === 'support' || type === 'tension') {
          const source = document.getElementById('insight-source').value;
          const target = document.getElementById('insight-target').value;
          if (source === target) { alert('Source and target must differ.'); return; }
          const weight = parseFloat(document.getElementById('insight-weight').value);
          const description = document.getElementById('insight-desc')?.value.trim() || '';
          const edge = { id: `e-${source}-${target}-${type}`, source, target, polarity: type, weight, description, contentPath: null };
          await patchEdge(edge);
          upsertEdge(edge);
        } else if (type === 'new-belief') {
          const label = document.getElementById('insight-label').value.trim();
          if (!label) { alert('Label required.'); return; }
          const id = slugify(label);
          const tier = document.getElementById('insight-tier')?.value || null;
          const confidence = document.getElementById('insight-confidence').value;
          const contentPath = `commitments/${id}.md`;
          const node = { id, label, type: 'domain', tier, confidence, contentPath, parentDomain: null, x: null, y: null };
          await patchNode(node);
          await saveFile(contentPath, `# ${label}\n\n**Confidence**: ${titleCase(confidence)}\n\n## Position\n\n*Captured from dialogue.*\n`);
          upsertNode(node);
        } else if (type === 'confidence') {
          const nodeId = document.getElementById('insight-node').value;
          const confidence = document.getElementById('insight-confidence').value;
          await patchNode({ id: nodeId, confidence });
          const n = graphData.nodes.find(x => x.id === nodeId);
          if (n) n.confidence = confidence;
        }
        const insightContainer = document.getElementById('capture-insight-container');
        if (insightContainer) insightContainer.innerHTML = '';
        alert('Insight captured.');
      } catch (err) { alert('Failed: ' + err.message); }
    });
  }

  // -----------------------------------------------------------------------
  // Readings
  // -----------------------------------------------------------------------

  async function renderReadings() {
    showLoading('Loading readings...');
    let files;
    try { files = await fetchFiles(); } catch { showError('Could not load files.'); return; }

    const readingFiles = filterFilesByDir(files, 'readings');
    let html = `<div class="page-header"><h1>Reading Notes</h1><p class="subtitle">Notes from your philosophical readings</p></div>`;

    // PDF upload dropzone
    html += `<div class="upload-dropzone" id="pdf-dropzone">
      <div class="dropzone-content" id="dropzone-content">
        <div class="dropzone-icon">+</div>
        <div class="dropzone-text">Drop a philosophical PDF here to generate a persona</div>
        <div class="dropzone-subtext">or <label for="pdf-file-input" class="dropzone-browse">browse files</label></div>
        <input type="file" id="pdf-file-input" accept=".pdf" style="display:none">
      </div>
      <div class="dropzone-progress" id="dropzone-progress" style="display:none">
        <div class="dropzone-spinner"></div>
        <div class="dropzone-progress-text" id="dropzone-progress-text">Extracting text and generating persona...</div>
      </div>
      <div class="dropzone-error" id="dropzone-error" style="display:none">
        <div class="dropzone-error-text" id="dropzone-error-text"></div>
        <button class="btn btn-small" id="dropzone-retry">Try Again</button>
      </div>
    </div>`;

    if (readingFiles.length === 0) {
      html += `<div class="empty-state"><h3>No reading notes yet</h3><p>Upload a PDF above or add markdown files to <code>readings/</code>.</p></div>`;
    } else {
      for (const f of readingFiles) {
        const path = f.path || f;
        const slug = fileBasename(path);
        html += `<div class="card card-clickable" onclick="window.location.hash='reading/${slug}'"><div class="card-title">${escapeHtml(titleCase(slug))}</div></div>`;
      }
    }
    setContent(html);

    // Wire up dropzone events
    const dropzone = document.getElementById('pdf-dropzone');
    const fileInput = document.getElementById('pdf-file-input');
    if (dropzone && fileInput) {
      dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
      dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.name.toLowerCase().endsWith('.pdf')) handlePdfUpload(file);
        else alert('Please drop a PDF file.');
      });
      fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) handlePdfUpload(fileInput.files[0]);
      });
      document.getElementById('dropzone-retry')?.addEventListener('click', () => {
        document.getElementById('dropzone-content').style.display = '';
        document.getElementById('dropzone-error').style.display = 'none';
      });
    }
  }

  async function handlePdfUpload(file) {
    const MAX_PDF_SIZE = 50 * 1024 * 1024;
    if (file.size > MAX_PDF_SIZE) {
      alert(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum is 50 MB.`);
      return;
    }
    const content = document.getElementById('dropzone-content');
    const progress = document.getElementById('dropzone-progress');
    const errorDiv = document.getElementById('dropzone-error');
    const errorText = document.getElementById('dropzone-error-text');
    const progressText = document.getElementById('dropzone-progress-text');

    content.style.display = 'none';
    errorDiv.style.display = 'none';
    progress.style.display = '';
    progressText.textContent = 'Extracting text from PDF...';

    // Cycle through progress stages on a timer
    const stages = [
      { text: 'Extracting text from PDF...', delay: 3000 },
      { text: 'Generating philosopher persona...', delay: 20000 },
      { text: 'Creating structured reading notes...', delay: 40000 },
      { text: 'Almost there — building analytical summaries...', delay: 70000 },
    ];
    let stageTimer = null;
    let stageIndex = 0;
    function advanceStage() {
      stageIndex++;
      if (stageIndex < stages.length) {
        progressText.textContent = stages[stageIndex].text;
        const nextDelay = (stages[stageIndex + 1] ? stages[stageIndex + 1].delay - stages[stageIndex].delay : 30000);
        stageTimer = setTimeout(advanceStage, nextDelay);
      }
    }
    stageTimer = setTimeout(advanceStage, stages[0].delay);

    try {
      const result = await uploadPdfForPersona(file);
      clearTimeout(stageTimer);
      progress.style.display = 'none';
      content.style.display = '';
      const name = result.persona ? result.persona.name : 'the philosopher';
      alert(`Persona generated for ${name}! Select them in the Dialogue view.`);
      invalidateCache();
      invalidatePersonasCache();
      renderReadings();
    } catch (err) {
      clearTimeout(stageTimer);
      progress.style.display = 'none';
      errorDiv.style.display = '';
      errorText.textContent = err.message || 'Upload failed. Please try again.';
    }
  }

  // -----------------------------------------------------------------------
  // Reading Detail
  // -----------------------------------------------------------------------

  async function renderReading(name) {
    if (!name) { navigate('readings'); return; }
    showLoading('Loading...');

    const hash = window.location.hash.slice(1);
    const qi = hash.indexOf('?');
    let dir = 'readings';
    if (qi !== -1) { const p = new URLSearchParams(hash.slice(qi + 1)).get('dir'); if (p) dir = p; }

    const path = `${dir}/${name}.md`;
    let data;
    try { data = await fetchFile(path); } catch { showError(`Could not load "${name}".`); return; }

    const content = data.content || '';
    const backHash = dir === 'dialogues' ? 'graph' : 'readings';
    const backLabel = dir === 'dialogues' ? 'Graph' : 'Reading Notes';
    let isEditing = false;

    function render() {
      let html = `<a href="#${backHash}" class="back-link">${backLabel}</a>
        <div class="flex-between mb-2"><h1>${escapeHtml(titleCase(name))}</h1>
        <button class="btn btn-small" id="edit-toggle">${isEditing ? 'Cancel' : 'Edit'}</button></div>`;

      if (isEditing) {
        html += `<div class="editor-container"><textarea class="editor-textarea" id="editor-content">${escapeHtml(content)}</textarea>
          <div class="editor-actions"><button class="btn" id="editor-cancel">Cancel</button><button class="btn btn-primary" id="editor-save">Save</button></div></div>`;
      } else {
        html += `<div class="md-content">${renderMarkdown(content)}</div>`;
      }
      setContent(html);

      document.getElementById('edit-toggle')?.addEventListener('click', () => { isEditing = !isEditing; render(); });
      document.getElementById('editor-cancel')?.addEventListener('click', () => { isEditing = false; render(); });
      document.getElementById('editor-save')?.addEventListener('click', async () => {
        const ta = document.getElementById('editor-content');
        if (!ta) return;
        try {
          document.getElementById('editor-save').disabled = true;
          document.getElementById('editor-save').textContent = 'Saving...';
          await saveFile(path, ta.value);
          data.content = ta.value;
          isEditing = false;
          renderReading(name);
        } catch (err) { alert('Failed: ' + err.message); }
      });
    }
    render();
  }

  // -----------------------------------------------------------------------
  // Not Found
  // -----------------------------------------------------------------------

  function renderNotFound() {
    setContent(`<div class="empty-state"><h3>Page not found</h3><p><a href="#graph">Return to graph.</a></p></div>`);
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  window.addEventListener('hashchange', () => {
    document.removeEventListener('keydown', handleGraphKeydown);
  });

  // -----------------------------------------------------------------------
  // Global API
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  // Belief Extraction (fire-and-forget + polling)
  // -----------------------------------------------------------------------

  function triggerBeliefExtraction(history, newTurnIndex) {
    const requestTime = Date.now() / 1000;

    fetch(`${API_BASE}/extract-beliefs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history, new_turn_index: newTurnIndex }),
    })
      .then(r => {
        if (r.status === 202) {
          pollExtractionResult(requestTime, 0);
        }
        // 200 = skipped, do nothing
      })
      .catch(err => console.warn('Belief extraction request failed:', err.message));
  }

  function pollExtractionResult(requestTime, attempt) {
    if (attempt >= 15) {
      console.warn('Extraction polling timed out — graph may update on next load');
      return;
    }

    const delay = Math.min(3000 * Math.pow(1.2, attempt), 10000);
    setTimeout(() => {
      fetch(`${API_BASE}/extract-beliefs/latest`)
        .then(r => r.json())
        .then(data => {
          if (data.timestamp && data.timestamp > requestTime && data.result) {
            applyExtractionDiff(data.result);
          } else {
            pollExtractionResult(requestTime, attempt + 1);
          }
        })
        .catch(err => console.warn('Extraction poll failed:', err.message));
    }, delay);
  }

  function applyExtractionDiff(diff) {
    const newNodes = diff.new_nodes || [];
    const newEdges = diff.new_edges || [];
    const confidenceUpdates = diff.confidence_updates || [];
    const labelUpdates = diff.label_updates || [];
    const edgeUpdates = diff.edge_updates || [];
    const edgeRemovals = diff.edge_removals || [];
    const retractedNodes = diff.retracted_nodes || [];
    const tensions = diff.tensions_identified || [];

    const totalOps = newNodes.length + newEdges.length + confidenceUpdates.length
      + labelUpdates.length + edgeUpdates.length + edgeRemovals.length
      + retractedNodes.length + tensions.length;
    if (totalOps === 0) return;

    // Merge into in-memory graphData
    if (graphData) {
      const existingNodeIds = new Set(graphData.nodes.map(n => n.id));
      for (const node of newNodes) {
        if (!existingNodeIds.has(node.id)) {
          graphData.nodes.push(node);
        }
      }
      const existingEdgeIds = new Set(graphData.edges.map(e => e.id));
      for (const edge of newEdges) {
        if (!existingEdgeIds.has(edge.id)) {
          graphData.edges.push(edge);
        }
      }
      for (const upd of confidenceUpdates) {
        const node = graphData.nodes.find(n => n.id === upd.node_id);
        if (node) node.confidence = upd.new_confidence;
      }
      for (const upd of labelUpdates) {
        const node = graphData.nodes.find(n => n.id === upd.node_id);
        if (node) node.label = upd.new_label;
      }
      for (const upd of edgeUpdates) {
        const edge = graphData.edges.find(e => e.id === upd.edge_id);
        if (edge && upd.new_weight !== undefined) edge.weight = upd.new_weight;
      }
      const removedEdgeIds = new Set(edgeRemovals.map(r => r.edge_id));
      if (removedEdgeIds.size > 0) {
        graphData.edges = graphData.edges.filter(e => !removedEdgeIds.has(e.id));
      }
      for (const ret of retractedNodes) {
        const node = graphData.nodes.find(n => n.id === ret.node_id);
        if (node) node.confidence = 'retracted';
      }
      // Tension edges are already in newEdges (added server-side),
      // but add any that weren't (defensive)
      for (const tens of tensions) {
        const edgeId = `e-${tens.between[0]}-${tens.between[1]}-tension`;
        if (!graphData.edges.find(e => e.id === edgeId)) {
          graphData.edges.push({
            id: edgeId,
            source: tens.between[0],
            target: tens.between[1],
            polarity: 'tension',
            weight: 0.5,
            description: tens.description || 'Identified tension',
          });
        }
      }

      // Redraw if graph view is currently active
      if (document.getElementById('graph-container')) {
        drawGraph();
      }
    }

    showExtractionToast({
      nodes: newNodes.length,
      edges: newEdges.length,
      confidenceUpdates: confidenceUpdates.length,
      labelUpdates: labelUpdates.length,
      edgeUpdates: edgeUpdates.length,
      edgeRemovals: edgeRemovals.length,
      retractions: retractedNodes.length,
      tensions: tensions.length,
    });
  }

  function showExtractionToast(counts) {
    const parts = [];
    if (counts.nodes > 0) parts.push(`${counts.nodes} belief${counts.nodes > 1 ? 's' : ''}`);
    if (counts.edges > 0) parts.push(`${counts.edges} connection${counts.edges > 1 ? 's' : ''}`);
    if (counts.confidenceUpdates > 0) parts.push(`${counts.confidenceUpdates} confidence update${counts.confidenceUpdates > 1 ? 's' : ''}`);
    if (counts.labelUpdates > 0) parts.push(`${counts.labelUpdates} label update${counts.labelUpdates > 1 ? 's' : ''}`);
    if (counts.edgeUpdates > 0) parts.push(`${counts.edgeUpdates} edge update${counts.edgeUpdates > 1 ? 's' : ''}`);
    if (counts.edgeRemovals > 0) parts.push(`${counts.edgeRemovals} edge${counts.edgeRemovals > 1 ? 's' : ''} removed`);
    if (counts.retractions > 0) parts.push(`${counts.retractions} retraction${counts.retractions > 1 ? 's' : ''}`);
    if (counts.tensions > 0) parts.push(`${counts.tensions} tension${counts.tensions > 1 ? 's' : ''} identified`);
    if (parts.length === 0) return;

    const msg = parts.join(', ') + ' extracted from dialogue';

    // Remove any existing toast
    const existing = document.querySelector('.extraction-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'extraction-toast';
    toast.innerHTML = `<span class="toast-icon">\u25C9</span> ${msg}`;
    document.body.appendChild(toast);

    // Trigger show animation
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });

    // Fade out after 4s
    setTimeout(() => {
      toast.classList.remove('show');
      toast.classList.add('fade-out');
      setTimeout(() => toast.remove(), 500);
    }, 4000);
  }

  window.app = {
    clearContext() {
      dialogueContext = null;
      dialogueContextContent = null;
      dialogueHistory = [];
      navigate('dialogue');
    },
    clearContest() {
      contestNode = null;
      contestSubgraph = null;
      contestMode = false;
      dialogueHistory = [];
      navigate('dialogue');
    },
  };

})();
