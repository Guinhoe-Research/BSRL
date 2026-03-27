from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json
import math
from pathlib import Path

app = FastAPI()

DATA_PATH = Path(__file__).resolve().parent.parent / "evals" / "eval_output.json"


def sanitize_floats(obj):
    """Replace inf/-inf/nan with JSON-safe values."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    return obj


with open(DATA_PATH) as f:
    raw = sanitize_floats(json.load(f))
    # The file is a list of rounds for a single game; wrap it as one episode.
    # If it's already a list of episodes (list of lists), use as-is.
    if raw and isinstance(raw[0], list):
        EPISODES = raw
    else:
        EPISODES = [raw]


@app.get("/api/episodes")
def list_episodes():
    summaries = []
    for i, ep in enumerate(EPISODES):
        total_steps = sum(len(r.get("steps", [])) for r in ep)
        agents = sorted({s["agent"] for r in ep for s in r.get("steps", [])})
        summaries.append({
            "index": i,
            "agents": agents,
            "total_steps": total_steps,
            "round_index": ep[-1].get("round_index") if ep else None,
        })
    return summaries


@app.get("/api/episodes/{ep_idx}")
def get_episode(ep_idx: int):
    rounds = EPISODES[ep_idx]
    # Flatten all rounds' steps into a single chronological timeline
    timeline = []
    for r in rounds:
        for s in r.get("steps", []):
            timeline.append(s)
    agents = sorted({s["agent"] for s in timeline})
    return {
        "index": ep_idx,
        "agents": agents,
        "game_state": {},
        "timeline": timeline,
    }


RANK_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BS Card Game — Eval Viewer</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3348; --text: #e0e0e6; --text2: #8b8fa3;
    --accent: #6c8cff; --accent2: #4a6adf;
    --green: #4ade80; --red: #f87171; --yellow: #fbbf24; --cyan: #22d3ee;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace; background:var(--bg); color:var(--text); }

  /* Layout */
  .app { display:flex; height:100vh; }
  .sidebar { width:280px; background:var(--surface); border-right:1px solid var(--border); display:flex; flex-direction:column; flex-shrink:0; }
  .sidebar-header { padding:16px; border-bottom:1px solid var(--border); }
  .sidebar-header h2 { font-size:14px; color:var(--accent); margin-bottom:8px; }
  .sidebar-header input { width:100%; padding:6px 10px; background:var(--surface2); border:1px solid var(--border); border-radius:6px; color:var(--text); font-size:12px; outline:none; }
  .sidebar-header input:focus { border-color:var(--accent); }
  .ep-list { flex:1; overflow-y:auto; padding:4px; }
  .ep-item { padding:8px 12px; border-radius:6px; cursor:pointer; font-size:12px; margin-bottom:2px; }
  .ep-item:hover { background:var(--surface2); }
  .ep-item.active { background:var(--accent2); color:#fff; }
  .ep-item .ep-meta { color:var(--text2); font-size:11px; }

  .main { flex:1; display:flex; flex-direction:column; overflow:hidden; }

  /* Top bar with step controls */
  .controls { display:flex; align-items:center; gap:12px; padding:12px 20px; background:var(--surface); border-bottom:1px solid var(--border); }
  .controls button { padding:6px 14px; background:var(--surface2); border:1px solid var(--border); border-radius:6px; color:var(--text); cursor:pointer; font-size:12px; font-family:inherit; }
  .controls button:hover:not(:disabled) { background:var(--accent2); border-color:var(--accent); }
  .controls button:disabled { opacity:0.3; cursor:default; }
  .step-label { font-size:13px; color:var(--text2); min-width:120px; text-align:center; }
  .step-slider { flex:1; accent-color:var(--accent); }
  .controls .autoplay-speed { width:60px; padding:4px 6px; background:var(--surface2); border:1px solid var(--border); border-radius:4px; color:var(--text); font-size:11px; text-align:center; }

  /* Content area */
  .content { flex:1; overflow-y:auto; padding:20px; display:grid; grid-template-columns:1fr 1fr; gap:16px; }

  .card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card h3 { font-size:12px; text-transform:uppercase; color:var(--text2); margin-bottom:12px; letter-spacing:1px; }

  /* Summary log */
  .log-panel { grid-column:1/3; }
  .log-entry { padding:6px 10px; border-radius:4px; font-size:12px; margin-bottom:4px; border-left:3px solid transparent; }
  .log-entry.current { background:var(--surface2); border-left-color:var(--accent); }
  .log-entry.past { opacity:0.5; }
  .log-entry .log-agent { font-weight:600; }
  .log-entry .log-agent.agent_0 { color:var(--cyan); }
  .log-entry .log-agent.agent_1 { color:var(--yellow); }
  .log-entry .log-agent.agent_2 { color:var(--green); }
  .log-entry .log-agent.agent_3 { color:var(--red); }

  /* Agent detail panels */
  .agent-panel { }
  .agent-panel.agent_0 { border-color: #22d3ee33; }
  .agent-panel.agent_1 { border-color: #fbbf2433; }

  /* Hand bar chart */
  .hand-chart { display:flex; align-items:flex-end; gap:3px; height:80px; margin-bottom:8px; }
  .hand-bar-col { display:flex; flex-direction:column; align-items:center; flex:1; }
  .hand-bar { width:100%; border-radius:3px 3px 0 0; min-height:2px; transition:height 0.2s; }
  .hand-bar.agent_0 { background:var(--cyan); }
  .hand-bar.agent_1 { background:var(--yellow); }
  .hand-bar.agent_2 { background:var(--green); }
  .hand-bar.agent_3 { background:var(--red); }
  .hand-label { font-size:10px; color:var(--text2); margin-top:4px; }

  /* Action mask */
  .mask-row { display:flex; gap:2px; margin-bottom:4px; flex-wrap:wrap; }
  .mask-cell { width:28px; height:22px; display:flex; align-items:center; justify-content:center; border-radius:3px; font-size:10px; }
  .mask-cell.on { background:#4ade8033; color:var(--green); }
  .mask-cell.off { background:#f8717133; color:var(--red); opacity:0.4; }
  .mask-cell.chosen { outline:2px solid var(--accent); outline-offset:1px; }

  /* Logits chart */
  .logits-chart { display:flex; align-items:flex-end; gap:2px; height:60px; }
  .logit-bar-col { display:flex; flex-direction:column; align-items:center; flex:1; }
  .logit-bar { width:100%; border-radius:2px 2px 0 0; background:var(--accent); min-height:1px; transition:height 0.2s; }
  .logit-bar.chosen { background:var(--green); }
  .logit-label { font-size:8px; color:var(--text2); margin-top:2px; }

  /* State vector */
  .state-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px,1fr)); gap:6px; }
  .state-item { display:flex; justify-content:space-between; font-size:11px; padding:4px 8px; background:var(--surface2); border-radius:4px; }
  .state-item .val { color:var(--accent); font-weight:600; }

  /* Claim sequence */
  .claim-seq { max-height:120px; overflow-y:auto; }
  .claim-row { font-size:11px; padding:2px 8px; }
  .claim-row.empty { color:var(--text2); opacity:0.3; }

  /* Responsive */
  @media(max-width:900px) {
    .content { grid-template-columns:1fr; }
    .log-panel { grid-column:1; }
  }

  /* Empty state */
  .empty-state { display:flex; align-items:center; justify-content:center; height:100%; color:var(--text2); font-size:14px; grid-column:1/3; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
</head>
<body>
<div class="app" id="app">
  <div class="sidebar">
    <div class="sidebar-header">
      <h2>BS Eval Viewer</h2>
      <input type="text" id="ep-search" placeholder="Search episodes..." oninput="filterEpisodes()">
    </div>
    <div class="ep-list" id="ep-list"></div>
  </div>
  <div class="main">
    <div class="controls" id="controls" style="display:none;">
      <button id="btn-prev" onclick="prevStep()">← Prev</button>
      <button id="btn-next" onclick="nextStep()">Next →</button>
      <button id="btn-play" onclick="togglePlay()">▶ Play</button>
      <input type="range" class="step-slider" id="step-slider" min="0" max="0" value="0" oninput="goToStep(+this.value)">
      <span class="step-label" id="step-label">Step 0 / 0</span>
      <input type="number" class="autoplay-speed" id="autoplay-speed" value="800" min="100" step="100" title="Autoplay interval (ms)">
    </div>
    <div class="content" id="content">
      <div class="empty-state">Select an episode from the sidebar</div>
    </div>
  </div>
</div>

<script>
const RANK_LABELS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"];
const ACTION_LABELS = [
  "Claim 1","Claim 2","Claim 3","Claim 4",
  ...RANK_LABELS.map(r=>"Sel "+r),
  "Challenge","Pass"
];

let episodes = [];
let currentEp = null;
let currentStep = 0;
let playInterval = null;

// --- Fetch episode list ---
async function init() {
  const res = await fetch("/api/episodes");
  episodes = await res.json();
  renderEpList();
}

function filterEpisodes() {
  renderEpList(document.getElementById("ep-search").value.trim());
}

function renderEpList(filter="") {
  const el = document.getElementById("ep-list");
  const items = episodes.filter(ep => {
    if (!filter) return true;
    return ("ep "+ep.index).includes(filter) || ep.agents.join(" ").includes(filter);
  });
  el.innerHTML = items.map(ep => `
    <div class="ep-item ${currentEp && currentEp.index===ep.index?'active':''}" onclick="loadEpisode(${ep.index})">
      <div>Episode ${ep.index}</div>
      <div class="ep-meta">${ep.agents.join(", ")} · ${ep.total_steps} steps · round ${ep.round_index}</div>
    </div>
  `).join("");
}

async function loadEpisode(idx) {
  stopPlay();
  const res = await fetch("/api/episodes/"+idx);
  currentEp = await res.json();
  currentStep = 0;
  document.getElementById("controls").style.display = "flex";
  const slider = document.getElementById("step-slider");
  slider.max = currentEp.timeline.length - 1;
  slider.value = 0;
  renderEpList();
  renderStep();
}

// --- Step navigation ---
function prevStep() { if(currentStep>0){ currentStep--; sync(); } }
function nextStep() { if(currentEp && currentStep<currentEp.timeline.length-1){ currentStep++; sync(); } }
function goToStep(s) { currentStep=s; renderStep(); }
function sync() {
  document.getElementById("step-slider").value = currentStep;
  renderStep();
}
function togglePlay() {
  if (playInterval) { stopPlay(); } else { startPlay(); }
}
function startPlay() {
  const speed = +document.getElementById("autoplay-speed").value || 800;
  document.getElementById("btn-play").textContent = "⏸ Pause";
  playInterval = setInterval(()=>{
    if(currentStep < currentEp.timeline.length-1){ currentStep++; sync(); }
    else { stopPlay(); }
  }, speed);
}
function stopPlay() {
  clearInterval(playInterval); playInterval=null;
  document.getElementById("btn-play").textContent = "▶ Play";
}

// --- Keyboard ---
document.addEventListener("keydown", e => {
  if (!currentEp) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); prevStep(); }
  if (e.key === "ArrowRight") { e.preventDefault(); nextStep(); }
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
});

// --- Render ---
function renderStep() {
  if (!currentEp) return;
  const tl = currentEp.timeline;
  const step = tl[currentStep];
  document.getElementById("step-label").textContent = `Step ${currentStep} / ${tl.length-1}`;
  document.getElementById("btn-prev").disabled = currentStep===0;
  document.getElementById("btn-next").disabled = currentStep===tl.length-1;

  const content = document.getElementById("content");
  content.innerHTML = "";

  // --- Log panel ---
  const logHtml = tl.map((s,i) => {
    const cls = i===currentStep?"current":i<currentStep?"past":"";
    return `<div class="log-entry ${cls}"><span class="log-agent ${s.agent}">${s.agent}</span> ${escHtml(s.summary)}</div>`;
  }).join("");
  content.innerHTML += `<div class="card log-panel"><h3>Action Log</h3>${logHtml}</div>`;

  // --- Per-agent panels ---
  // Collect latest state for each agent up to current step
  const agentLatest = {};
  for (let i=0; i<=currentStep; i++) {
    agentLatest[tl[i].agent] = tl[i];
  }

  for (const agent of currentEp.agents) {
    const s = agentLatest[agent];
    if (!s) {
      content.innerHTML += `<div class="card agent-panel ${agent}"><h3>${agent}</h3><div style="color:var(--text2);font-size:12px;">No actions yet</div></div>`;
      continue;
    }
    const isActive = s === step && agent === step.agent;
    let html = `<div class="card agent-panel ${agent}" ${isActive?'style="border-color:var(--accent);"':''}>`;
    html += `<h3>${agent} ${isActive?'(current)':''}</h3>`;

    // Hand chart
    const handCounts = decodeHand(s.state);
    html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--text2);margin-bottom:4px;">Hand (${handCounts.reduce((a,b)=>a+b,0)} cards)</div>`;
    html += `<div class="hand-chart">`;
    for (let r=0;r<13;r++) {
      const h = handCounts[r] * 25;
      html += `<div class="hand-bar-col"><div class="hand-bar ${agent}" style="height:${h}px;"></div><div class="hand-label">${RANK_LABELS[r]}</div></div>`;
    }
    html += `</div></div>`;

    // Action mask + chosen action
    html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--text2);margin-bottom:4px;">Action Mask (chosen: ${ACTION_LABELS[s.action]})</div>`;
    html += `<div class="mask-row">`;
    for (let a=0;a<19;a++) {
      const on = s.action_mask[a]===1;
      const chosen = a===s.action;
      html += `<div class="mask-cell ${on?'on':'off'} ${chosen?'chosen':''}" title="${ACTION_LABELS[a]}">${ACTION_LABELS[a].replace("Claim ","C").replace("Sel ","").replace("Challenge","CH").replace("Pass","PA")}</div>`;
    }
    html += `</div></div>`;

    // Logits (softmax probabilities)
    if (s.logits) {
      const probs = softmax(s.logits);
      const maxP = Math.max(...probs);
      html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--text2);margin-bottom:4px;">Policy (softmax) · log_prob: ${s.log_prob?.toFixed(3) ?? 'N/A'}</div>`;
      html += `<div class="logits-chart">`;
      for (let a=0;a<19;a++) {
        const h = maxP>0 ? (probs[a]/maxP)*55 : 0;
        const chosen = a===s.action;
        html += `<div class="logit-bar-col"><div class="logit-bar ${chosen?'chosen':''}" style="height:${h}px;" title="${ACTION_LABELS[a]}: ${(probs[a]*100).toFixed(1)}%"></div><div class="logit-label">${(probs[a]*100).toFixed(0)}</div></div>`;
      }
      html += `</div></div>`;
    }

    // State vector breakdown
    html += renderStateBreakdown(s.state, agent);

    html += `</div>`;
    content.innerHTML += html;
  }

  // Scroll log to current
  const logEntries = document.querySelectorAll(".log-entry.current");
  if (logEntries.length) logEntries[0].scrollIntoView({block:"nearest"});
}

function decodeHand(state) {
  // state[3:16] = hand counts normalized by /4.0
  return state.slice(3,16).map(v => Math.round(v*4));
}

function renderStateBreakdown(state, agent) {
  const phaseIdx = state.slice(0,3).indexOf(1.0);
  const phaseNames = ["CLAIM","SELECT","CHALLENGE"];
  const hand = state.slice(3,16).map(v=>(v*4).toFixed(1));
  const pile = (state[16]*52).toFixed(0);
  const claimRank = Math.round(state[17]*13);
  const lastClaimRank = Math.round(state[18]*13);
  const lastClaimCount = Math.round(state[19]*4);

  let html = `<div style="margin-bottom:8px;"><div style="font-size:11px;color:var(--text2);margin-bottom:4px;">State Breakdown</div>`;
  html += `<div class="state-grid">`;
  html += `<div class="state-item"><span>Phase</span><span class="val">${phaseNames[phaseIdx]||'?'}</span></div>`;
  html += `<div class="state-item"><span>Pile Size</span><span class="val">${pile}</span></div>`;
  html += `<div class="state-item"><span>Claim Rank</span><span class="val">${claimRank>0?RANK_LABELS[claimRank-1]:'—'}</span></div>`;
  html += `<div class="state-item"><span>Last Claim</span><span class="val">${lastClaimRank>0?lastClaimCount+'× '+RANK_LABELS[lastClaimRank-1]:'—'}</span></div>`;
  // Card counts (indices 20+)
  if (state.length > 20) {
    for (let i=20;i<state.length;i++) {
      html += `<div class="state-item"><span>agent_${i-20} cards</span><span class="val">${Math.round(state[i]*52)}</span></div>`;
    }
  }
  html += `</div></div>`;
  return html;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(l=>Math.exp(l-max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(e=>e/sum);
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
