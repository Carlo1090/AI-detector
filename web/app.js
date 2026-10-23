"use strict";

// Example texts live in samples.js (SAMPLES.ai / .human / .mixed).

const VERDICTS = {
  likely_ai: "Likely AI-generated",
  mixed: "Mix of AI and human writing",
  likely_human: "Likely human-written",
  too_short: "Too short to judge",
  uncalibrated: "Model not calibrated",
};

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), statusText: $("status-text"), input: $("input"), doc: $("doc"), tabs: $("tabs"), htabs: $("htabs"),
  title: $("pane-title"), words: $("word-count"), edit: $("edit"), analyze: $("analyze"),
  batch: $("batch"), files: $("files"), batchClose: $("batch-close"), batchAdd: $("batch-add"), batchCsv: $("batch-csv"),
  toBatch: $("to-batch"), bresults: $("bresults"), batchBar: $("batch-bar"), batchStatus: $("batch-status"),
  batchFacts: $("batch-facts"), live: $("live"), explain: $("verdict-explain"), again: $("again"),
  sensitivity: $("sensitivity"), sensNote: $("sens-note"),
  upload: $("upload"), file: $("file"), toHumanize: $("to-humanize"), humanize: $("humanize"), print: $("print"),
  strength: $("strength"), tone: $("tone"), humanizeNote: $("humanize-note"),
  backReport: $("back-report"), download: $("download"), recheck: $("recheck"), copy: $("copy"),
  empty: $("empty"), history: $("history"), historyList: $("history-list"), clearHistory: $("clear-history"),
  loading: $("loading"), loadingText: $("loading-text"), loadingTime: $("loading-time"), error: $("error"),
  results: $("results"), hresults: $("hresults"), ringFill: $("ring-fill"), strip: $("strip"), stripCounts: $("strip-counts"),
  share: $("share"), shareCaption: $("share-caption"), verdict: $("verdict"), verdictDetail: $("verdict-detail"),
  manipulation: $("manipulation"), signals: $("signals"), legend: $("legend"), tip: $("tip"),
  hBefore: $("h-before"), hAfter: $("h-after"), hBeforeV: $("h-before-v"), hAfterV: $("h-after-v"), hFacts: $("h-facts"),
  pop: $("pop"), popScore: $("pop-score"), popSentence: $("pop-sentence"), popReasons: $("pop-reasons"),
  popSuggest: $("pop-suggest"), popOptions: $("pop-options"), popClose: $("pop-close"),
  theme: $("theme"), accuracy: $("accuracy"), accuracyBody: $("accuracy-body"),
};

const state = {
  ready: false, busy: false, humanizer: "waiting", mode: "edit", model: "",
  result: null, text: "", view: "sentences",
  humanized: null, accepted: new Set(), hview: "changes",
  pop: null,  // index of the sentence whose popover is open
};

const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const countWords = (t) => (t.match(/[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*/g) || []).length;
const show = (el, on) => { el.hidden = !on; };
const segValue = (group) => group.querySelector('[aria-checked="true"]').dataset.v;

const store = {  // browser storage can be unavailable (private windows, blocked site data)
  get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* not critical */ } },
};

/* ---------- sensitivity ---------- */

const SENS_NOTES = {
  strict: "Fewest false alarms: tuned so about 1 in 100 human-written texts gets flagged. Misses more AI, especially from the newest models.",
  balanced: "Catches more AI; about 2–3 in 100 human-written texts may be flagged.",
  sensitive: "Catches the most AI, including newer models, but about 5 in 100 human-written texts may be flagged. Use it to screen, not to judge.",
};
state.sensitivity = store.get("detectai.sensitivity", "strict");

// Re-derive flags, share and verdict for another sensitivity from the probabilities already in the report.
// Mirrors Detector._verdict on the server.
function rethreshold(r, text, level) {
  const cut = r.levels?.[level];
  if (!cut) return r;
  let flaggedWords = 0;
  for (const s of r.sentences) {
    s.flagged = !r.too_short && s.probability != null && s.probability >= cut.window;
    if (s.flagged) flaggedWords += countWords(text.slice(s.start, s.end));
  }
  r.sensitivity = level;
  if (r.too_short || r.document_probability == null) return r;
  r.ai_share = r.words ? flaggedWords / r.words : 0;
  const docAI = r.document_probability >= cut.document;
  r.verdict = docAI && r.ai_share >= 0.5 ? "likely_ai" : docAI || r.ai_share >= 0.2 ? "mixed" : "likely_human";
  return r;
}

function syncSensitivity() {
  const available = state.result?.levels || {};
  for (const b of els.sensitivity.querySelectorAll("button")) {
    b.setAttribute("aria-checked", String(b.dataset.v === state.sensitivity));
    b.disabled = state.result != null && !available[b.dataset.v];
  }
  els.sensNote.textContent = SENS_NOTES[state.sensitivity];
}

els.sensitivity.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-v]");
  if (!b || b.disabled) return;
  state.sensitivity = b.dataset.v;
  store.set("detectai.sensitivity", state.sensitivity);
  for (const x of state.batch || []) if (x.result) rethreshold(x.result, x.text, state.sensitivity);
  if (state.result) { rethreshold(state.result, state.text, state.sensitivity); renderReport(); }
  live.text = null;  // the live score was computed at the old level
});

/* ---------- theme ---------- */

const THEMES = ["system", "light", "dark"];
function applyTheme(theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  els.theme.title = `Theme: ${theme}`;
  try { localStorage.setItem("detectai.theme", theme); } catch { /* ignore */ }
}
els.theme.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme || "system";
  applyTheme(THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
});
els.theme.title = `Theme: ${document.documentElement.dataset.theme || "system"}`;

/* ---------- model status ---------- */

async function pollHealth() {
  let again = true;
  try {
    const h = await (await fetch("/api/health")).json();
    els.status.dataset.state = h.status;
    if (h.status === "ready") {
      const first = !state.ready;
      state.ready = true;
      state.humanizer = h.humanizer;
      state.model = h.model;
      const rw = h.humanizer === "ready" ? " · humanizer ready" : h.humanizer === "loading" ? " · humanizer loading…" : "";
      els.statusText.textContent = `Detector ready · ${h.device.toUpperCase()}${h.calibrated ? "" : " · uncalibrated"}${rw}`;
      els.statusText.title = h.model + (h.rewriter ? ` · rewriter ${h.rewriter}` : "");
      syncButtons();
      if (first) runDemoFromHash();
      again = h.humanizer === "loading" || h.humanizer === "waiting";
    } else if (h.status === "error") {
      els.statusText.textContent = `Model failed to load: ${h.error}`;
      again = false;
    } else {
      els.statusText.textContent = "Loading language models… (first run downloads them)";
    }
  } catch {
    els.status.dataset.state = "error";
    els.statusText.textContent = "Server not reachable. Is it running?";
  }
  if (again) setTimeout(pollHealth, 2000);
}

function syncButtons() {
  const n = countWords(els.input.value);
  if (state.mode === "edit") {
    els.words.textContent = `${n.toLocaleString()} word${n === 1 ? "" : "s"}${n > 0 && n < 150 ? " · 150+ recommended" : ""}`;
  }
  els.analyze.disabled = !state.ready || state.busy || n === 0;
  const strength = segValue(els.strength);
  const needsModel = strength !== "light";
  const modelReady = state.humanizer === "ready";
  els.humanize.disabled = state.busy || (needsModel && !modelReady);
  els.toHumanize.disabled = state.busy;
  els.humanizeNote.textContent = needsModel && !modelReady
    ? (state.humanizer === "loading" ? "The rewriting model is still loading. Light mode works now."
      : `The rewriting model isn't available (${state.humanizer}). Light mode works.`)
    : strength === "light" ? "Swaps AI-typical words and phrases. Instant, no rewriting."
    : strength === "strong" ? "Rewrites anything borderline, up to three rounds. Slower."
    : "Rewrites flagged passages, checks the meaning, keeps the version that reads most human.";
}

function setMode(mode) {
  state.mode = mode;
  for (const group of document.querySelectorAll(".actions[data-mode]")) show(group, group.dataset.mode === mode);
  show(els.input, mode === "edit");
  show(els.doc, mode !== "edit");
  show(els.tabs, mode === "report");
  show(els.htabs, mode === "humanized");
  show(els.results, mode === "report");
  show(els.hresults, mode === "humanized");
  show(els.bresults, mode === "batch");
  show(els.toBatch, mode === "report" && state.batch != null);
  show(els.empty, mode === "edit");
  if (mode !== "edit") els.live.textContent = "";
  renderHistory();
  els.title.textContent = {
    edit: "Check a text for AI writing", report: "Report", humanized: "Humanized text",
    batch: `Batch check · ${state.batch ? state.batch.length : 0} files`,
  }[mode];
  hideTip();
  closePop();
}

let loadingTimer = null;
async function post(url, body, stages) {
  state.busy = true;
  syncButtons();
  show(els.error, false);
  const t0 = performance.now();
  const tick = () => {
    const s = (performance.now() - t0) / 1000;
    els.loadingText.textContent = [...stages].reverse().find(([at]) => s >= at)[1];
    els.loadingTime.textContent = s >= 2 ? `${Math.floor(s)} s` : "";
  };
  tick();
  loadingTimer = setInterval(tick, 500);
  show(els.loading, true);
  try {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The server rejected the request.");
    return data;
  } catch (err) {
    els.error.textContent = err.message || String(err);
    show(els.error, true);
    return null;
  } finally {
    clearInterval(loadingTimer);
    state.busy = false;
    show(els.loading, false);
    syncButtons();
  }
}

/* ---------- history (this browser only) ---------- */

function remember(r, text) {
  const items = store.get("detectai.history", []).filter((h) => h.text !== text);
  items.unshift({ t: Date.now(), words: r.words, share: r.ai_share, verdict: r.verdict, text: text.slice(0, 30000) });
  store.set("detectai.history", items.slice(0, 10));
}

function renderHistory() {
  const items = store.get("detectai.history", []);
  show(els.history, state.mode === "edit" && items.length > 0);
  els.historyList.innerHTML = items.map((h, i) => {
    const when = new Date(h.t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    const snippet = h.text.replace(/\s+/g, " ").slice(0, 70);
    return `<li><button type="button" data-i="${i}">
      <span class="h-snippet">${esc(snippet)}</span><span class="h-score">${h.share == null ? "–" : pct(h.share)}</span>
      <span class="h-meta">${esc(when)} · ${h.words} words · ${esc(VERDICTS[h.verdict] || "")}</span></button></li>`;
  }).join("");
}

els.historyList.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-i]");
  if (!b) return;
  const item = store.get("detectai.history", [])[Number(b.dataset.i)];
  if (item) { els.input.value = item.text; syncButtons(); analyze(item.text); }
});
els.clearHistory.addEventListener("click", () => { store.set("detectai.history", []); renderHistory(); });

/* ---------- detection report ---------- */

const CHECK_STAGES = [[0, "Reading your text word by word…"], [4, "Scoring each sentence with its neighbours…"]];

/* Live score: a quiet check after you stop typing, so the footer shows the AI score as you write and
   "Check text" opens instantly when nothing changed since. */
const live = { timer: null, seq: 0, text: null, result: null };

function scheduleLive() {
  clearTimeout(live.timer);
  const text = els.input.value;
  if (!state.ready || state.mode !== "edit" || countWords(text) < 150) { els.live.textContent = ""; return; }
  if (live.text === text) return;
  live.timer = setTimeout(async () => {
    const seq = ++live.seq;
    els.live.className = "live";
    els.live.textContent = "· scoring…";
    try {
      const res = await fetch("/api/detect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, sensitivity: state.sensitivity }) });
      const r = await res.json();
      if (!res.ok || seq !== live.seq || els.input.value !== text) return;
      live.text = text;
      live.result = r;
      const cls = r.verdict === "likely_ai" ? "ai" : r.verdict === "mixed" ? "mid" : "hu";
      els.live.className = `live ${cls}`;
      els.live.textContent = `· live: ${pct(r.ai_share)} flagged as AI`;
    } catch { if (seq === live.seq) els.live.textContent = ""; }
  }, 1500);
}

async function analyze(text = els.input.value) {
  if (!text.trim() || !state.ready || state.busy) return;
  show(els.empty, false);
  show(els.history, false);
  const cached = live.text === text ? live.result : null;
  const r = cached || await post("/api/detect", { text, sensitivity: state.sensitivity }, CHECK_STAGES);
  if (!r) { setMode(state.mode); return; }
  state.result = r;
  state.text = text;
  remember(r, text);
  renderReport();
}

// One plain-language sentence saying why, built from the signals that pushed the score the most.
const PLAIN = {
  probe: ["its style matches AI-written texts", "its style matches human-written texts"],
  binoculars: ["a chat model finds it unusually unsurprising", "a chat model finds it surprising, as with human text"],
  curvature: ["its word choices are unusually typical", "its word choices are less predictable than AI's"],
  mean_logprob: ["the wording is very predictable", "the wording is less predictable"],
  top10_frac: ["most words are a model's top guesses", "many words are ones a model wouldn't guess"],
  burst_ppl: ["sentences are evenly paced", "sentence rhythm varies the way people's does"],
  burst_len: ["sentence lengths are uniform", "sentence lengths vary"],
};

function explainVerdict(r) {
  if (r.too_short || r.verdict === "uncalibrated") return "";
  const leanAI = r.verdict !== "likely_human";
  const reasons = r.signals
    .filter((s) => (leanAI ? s.contribution > 0.4 : s.contribution < -0.4))
    .map((s) => (s.key === "lexicon"
      ? (leanAI && r.patterns.length ? `it uses ${r.patterns.length} AI-typical words and phrases` : null)
      : PLAIN[s.key]?.[leanAI ? 0 : 1]))
    .filter(Boolean).slice(0, 3);
  if (!reasons.length) return "";
  const list = reasons.length > 1 ? `${reasons.slice(0, -1).join(", ")} and ${reasons.at(-1)}` : reasons[0];
  const lead = { likely_ai: "Reads like AI", mixed: "Parts read like AI", likely_human: "Reads like a person wrote it" }[r.verdict];
  return `${lead}: ${list}.`;
}

function sentenceClass(s) {
  if (s.probability == null) return "na";
  if (s.flagged) return "ai";
  return s.probability >= 0.4 ? "mid" : "hu";
}

function renderReport() {
  const r = state.result;
  const share = r.ai_share;
  els.share.textContent = pct(share);
  els.shareCaption.textContent = share == null ? "not scored" : "flagged as AI";
  els.ringFill.style.strokeDashoffset = String(326.7 * (1 - (share ?? 0)));

  els.verdict.textContent = VERDICTS[r.verdict] || r.verdict;
  els.verdict.dataset.v = r.verdict;
  syncSensitivity();
  els.explain.textContent = explainVerdict(r);
  show(els.explain, els.explain.textContent !== "");
  const flagged = r.sentences.filter((s) => s.flagged).length;
  els.verdictDetail.textContent = r.too_short
    ? `Add more text. At least ${r.min_words} words are needed for a reliable score; this has ${r.words}. The word view still works.`
    : `${flagged} of ${r.sentences.length} sentences flagged. Whole-document AI probability: ${pct(r.document_probability)}.`;

  // Turnitin-style sentence map: one segment per sentence, width by length, colour by likelihood.
  const counts = { ai: 0, mid: 0, hu: 0, na: 0 };
  els.strip.innerHTML = r.sentences.map((s, i) => {
    const cls = sentenceClass(s);
    counts[cls] += 1;
    const w = Math.max(1, countWords(state.text.slice(s.start, s.end)));
    const label = s.probability == null ? "not scored" : `${pct(s.probability)} AI likelihood`;
    return `<button type="button" role="listitem" class="${cls}" style="--w:${w}" data-i="${i}" title="Sentence ${i + 1}: ${label}"></button>`;
  }).join("");
  els.stripCounts.textContent = r.too_short ? "too short to flag"
    : `${counts.ai} AI · ${counts.mid} unsure · ${counts.hu} human`;

  const m = r.manipulation || { homoglyphs: 0, invisible: 0 };
  const tricks = [];
  if (m.homoglyphs) tricks.push(`${m.homoglyphs} look-alike letter${m.homoglyphs === 1 ? "" : "s"} from other alphabets`);
  if (m.invisible) tricks.push(`${m.invisible} invisible character${m.invisible === 1 ? "" : "s"}`);
  els.manipulation.innerHTML = tricks.length
    ? `<b>Hidden characters found:</b> ${esc(tricks.join(" and "))}. These are a common trick to fool detectors and plagiarism checkers; the text was analysed with them undone.`
    : "";
  show(els.manipulation, tricks.length > 0);

  els.signals.innerHTML = r.signals.map((s) => {
    const c = Math.max(-1, Math.min(1, s.contribution / 3));
    const w = Math.abs(c) * 50;
    const side = c >= 0 ? `left:50%;width:${w}%;background:rgb(var(--ai))` : `left:${50 - w}%;width:${w}%;background:rgb(var(--human))`;
    return `<li class="signal">
      <div class="signal-top"><b>${esc(s.label)}</b><span>${esc(s.value)}</span></div>
      <div class="bar" title="${c >= 0 ? "Leans AI" : "Leans human"}"><i style="${side}"></i></div>
      <p>${esc(s.description)}</p></li>`;
  }).join("");

  els.words.textContent = `${r.words.toLocaleString()} words · checked in ${(r.elapsed_ms / 1000).toFixed(1)}s`;
  $("print-date").textContent = new Date().toLocaleString();  // header of the printed / PDF report
  $("print-model").textContent = r.model;
  setMode("report");
  setView(state.view);
}

// Hovering the sentence map outlines that sentence in the document.
function setHot(i) {
  for (const m of els.doc.querySelectorAll("mark.hot")) m.classList.remove("hot");
  if (i != null && state.view === "sentences") els.doc.querySelector(`mark[data-k="s"][data-i="${i}"]`)?.classList.add("hot");
}
els.strip.addEventListener("mouseover", (e) => setHot(e.target.closest("button[data-i]")?.dataset.i ?? null));
els.strip.addEventListener("mouseleave", () => setHot(null));

els.strip.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-i]");
  if (!b) return;
  if (state.view !== "sentences") setView("sentences");
  const mark = els.doc.querySelector(`mark[data-k="s"][data-i="${b.dataset.i}"]`);
  if (!mark) return;
  mark.scrollIntoView({ behavior: "smooth", block: "center" });
  mark.classList.remove("flash");
  void mark.offsetWidth;  // restart the animation
  mark.classList.add("flash");
});

function segmentsHtml(text, segs) {
  let out = "", pos = 0;
  for (const g of segs) {
    if (g.start < pos || g.end <= g.start) continue;
    out += esc(text.slice(pos, g.start));
    out += `<mark class="${g.cls}" data-k="${g.k}" data-i="${g.i}"${g.style ? ` style="${g.style}"` : ""}>${esc(text.slice(g.start, g.end))}</mark>`;
    pos = g.end;
  }
  return out + esc(text.slice(pos));
}

const rankClass = (r) => (r === 1 ? "r1" : r <= 10 ? "r10" : r <= 100 ? "r100" : "rx");

const VIEWS = {
  sentences(r) {
    return r.sentences.map((s, i) => {
      const a = s.flagged ? Math.min(0.34, 0.1 + 0.48 * Math.max(0, s.probability - 0.5)) : 0;
      return { start: s.start, end: s.end, i, k: "s", cls: s.flagged ? "s" : "s off", style: `--a:${a.toFixed(3)}` };
    });
  },
  words(r) {
    return r.tokens.map((t, i) => ({ start: t.s, end: t.e, i, k: "w", cls: `w ${rankClass(t.r)}` }));
  },
  phrases(r) {
    return r.patterns.map((p, i) => ({ start: p.start, end: p.end, i, k: "p", cls: p.weight < 1 ? "p light" : "p" }));
  },
};

const swatch = (bg, extra = "") => `<span class="swatch" style="background:${bg};${extra}"></span>`;

const LEGENDS = {
  sentences: () => `<h2>Sentence view</h2>
    <div class="legend-row">${swatch("rgb(var(--ai) / 0.4)", "box-shadow: inset 0 -2px 0 rgb(var(--ai))")}Flagged as likely AI (darker = more confident)</div>
    <div class="legend-row">${swatch("transparent")}Not flagged</div>
    <p class="muted small" style="margin:4px 0 0">Each sentence is judged together with its neighbours. Click one for details and rewrite suggestions.</p>`,
  words: () => `<h2>Word view</h2>
    <div class="legend-row">${swatch("rgb(var(--ai) / 0.42)")}The model's #1 guess</div>
    <div class="legend-row">${swatch("rgb(var(--ai) / 0.2)")}In its top 10 guesses</div>
    <div class="legend-row">${swatch("rgb(var(--ai) / 0.07)")}In its top 100</div>
    <div class="legend-row">${swatch("rgb(var(--human) / 0.22)")}Surprising word (human-like)</div>
    <p class="muted small" style="margin:4px 0 0">Long orange runs are what makes text read as AI. Hover a word to see what the model expected instead.</p>`,
  phrases: (r) => `<h2>AI phrases · ${r.patterns.length} found</h2>
    <div class="legend-row">${swatch("rgb(var(--ai) / 0.12)", "border-bottom: 2px solid rgb(var(--ai))")}AI-typical word, phrase or sentence shape</div>
    <div class="legend-row">${swatch("transparent", "border-bottom: 2px dotted rgb(var(--ai))")}Weak on its own, e.g. em dashes</div>
    <p class="muted small" style="margin:4px 0 0">The humanizer rewrites these.</p>`,
};

function setView(view) {
  state.view = view;
  for (const b of els.tabs.querySelectorAll("button")) b.setAttribute("aria-selected", String(b.dataset.view === view));
  els.doc.innerHTML = segmentsHtml(state.text, VIEWS[view](state.result));
  els.legend.innerHTML = LEGENDS[view](state.result);
  hideTip();
  closePop();
}

/* ---------- sentence popover: details + rewrite suggestions ---------- */

function openPop(i, x, y) {
  const s = state.result.sentences[i];
  state.pop = i;
  els.popScore.textContent = s.probability == null ? "Not scored"
    : `${pct(s.probability)} AI likelihood${s.flagged ? " · flagged" : ""}`;
  const sentence = state.text.slice(s.start, s.end);
  els.popSentence.textContent = sentence.length > 320 ? `${sentence.slice(0, 320)}…` : sentence;
  els.popReasons.innerHTML = s.reasons.map((x) => `<li>${esc(x)}</li>`).join("");
  els.popOptions.innerHTML = "";
  els.popOptions.className = "pop-options";
  els.popSuggest.disabled = state.busy;
  els.popSuggest.textContent = "Suggest rewrites";
  show(els.popSuggest, true);
  show(els.pop, true);
  hideTip();
  const { offsetWidth: w, offsetHeight: h } = els.pop;
  els.pop.style.left = `${Math.max(12, Math.min(x - w / 2, window.innerWidth - w - 12))}px`;
  els.pop.style.top = `${y + 16 + h > window.innerHeight ? Math.max(12, y - h - 16) : y + 16}px`;
}

function closePop() {
  state.pop = null;
  show(els.pop, false);
}

async function suggest() {
  const i = state.pop;
  if (i == null || state.busy) return;
  const s = state.result.sentences[i];
  els.popSuggest.disabled = true;
  els.popSuggest.textContent = "Writing and checking rewrites… (about 10–20 s)";
  state.busy = true;
  syncButtons();
  let data = null;
  try {
    const res = await fetch("/api/rewrite", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: state.text, start: s.start, end: s.end, tone: segValue(els.tone) }),
    });
    data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Couldn't get suggestions.");
  } catch (err) {
    els.popOptions.innerHTML = `<p class="muted">${esc(err.message || String(err))}</p>`;
    data = null;
  } finally {
    state.busy = false;
    syncButtons();
  }
  if (state.pop !== i) return;  // closed while waiting
  show(els.popSuggest, false);
  if (!data) return;
  if (!data.options.length) {
    els.popOptions.innerHTML = `<p class="muted">No rewrite passed the meaning and quality checks. Try again, or switch tone.</p>`;
    show(els.popSuggest, true);
    els.popSuggest.disabled = false;
    els.popSuggest.textContent = "Try again";
    return;
  }
  els.popOptions.innerHTML = data.options.map((o, k) => {
    const better = o.ai < data.ai;
    return `<div class="option">
      <p>${esc(o.text)}</p>
      <div class="option-foot">
        <span class="badge ${o.ai < 0.5 ? "good" : "bad"}" title="AI likelihood of this version">AI ${pct(o.ai)}${better ? "" : " (no better)"}</span>
        <span class="badge" title="How closely the meaning matches the original">${o.kind === "rule" ? "Word fixes" : `Meaning ${pct(o.similarity)}`}</span>
        <button class="btn ghost small" type="button" data-copy="${k}">Copy</button>
        <button class="btn primary small" type="button" data-use="${k}">Use</button>
      </div></div>`;
  }).join("");
  els.popOptions.dataset.options = JSON.stringify(data.options.map((o) => o.text));
}

els.popOptions.addEventListener("click", async (e) => {
  const options = JSON.parse(els.popOptions.dataset.options || "[]");
  const copy = e.target.closest("button[data-copy]");
  if (copy) {
    try { await navigator.clipboard.writeText(options[Number(copy.dataset.copy)]); copy.textContent = "Copied ✓"; } catch { copy.textContent = "Can't copy"; }
    return;
  }
  const b = e.target.closest("button[data-use]");
  if (!b || state.pop == null) return;
  const s = state.result.sentences[state.pop];
  const text = state.text.slice(0, s.start) + options[Number(b.dataset.use)] + state.text.slice(s.end);
  closePop();
  els.input.value = text;
  analyze(text);
});
els.popSuggest.addEventListener("click", suggest);
els.popClose.addEventListener("click", closePop);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closePop(); hideTip(); } });
document.addEventListener("mousedown", (e) => {
  if (!els.pop.hidden && !els.pop.contains(e.target) && !e.target.closest('mark[data-k="s"]')) closePop();
});

/* ---------- batch check: a whole class set of essays ---------- */

function startBatch(fileList) {
  const files = [...fileList];
  if (!files.length || !state.ready) return;
  state.batch = state.batch || [];
  for (const f of files) state.batch.push({ name: f.name, file: f, text: null, result: null, error: null, status: "queued" });
  setMode("batch");
  renderBatch();
  if (!state.batchRunning) runBatch();
}

async function runBatch() {
  state.batchRunning = true;
  const items = state.batch;
  for (const item of items) {
    if (item.status !== "queued") continue;
    item.status = "checking";
    renderBatch();
    try {
      item.text = (await readTextFile(item.file)).slice(0, 30000);
      if (!item.text.trim()) throw new Error("No text found in this file.");
      const res = await fetch("/api/detect", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: item.text, sensitivity: state.sensitivity }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The server rejected this file.");
      item.result = data;
      item.status = "done";
    } catch (err) {
      item.error = err.message || String(err);
      item.status = "error";
    }
    item.file = null;
    renderBatch();
  }
  state.batchRunning = false;
  if (state.batch && state.batch.some((x) => x.status === "queued")) runBatch();  // files added meanwhile
}

function renderBatch() {
  const items = state.batch;
  if (!items) return;
  const done = items.filter((x) => x.status === "done");
  const finished = items.filter((x) => x.status === "done" || x.status === "error").length;
  const checking = items.findIndex((x) => x.status === "checking");
  els.batchBar.style.width = `${(100 * finished) / items.length}%`;
  els.batchStatus.textContent = finished === items.length ? `All ${items.length} files checked.`
    : `Checking file ${checking + 1} of ${items.length}…`;
  const count = (v) => done.filter((x) => x.result.verdict === v).length;
  const shares = done.map((x) => x.result.ai_share).filter((x) => x != null);
  const facts = [
    ["Likely AI-generated", count("likely_ai")], ["Mixed", count("mixed")], ["Likely human", count("likely_human")],
    ["Too short to judge", count("too_short")], ["Couldn't read", items.filter((x) => x.status === "error").length],
    ["Average share flagged", shares.length ? pct(shares.reduce((a, b) => a + b, 0) / shares.length) : "–"],
  ];
  els.batchFacts.innerHTML = facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("");
  if (state.mode !== "batch") return;
  els.title.textContent = `Batch check · ${items.length} file${items.length === 1 ? "" : "s"}`;
  const rows = items.map((x, i) => {
    if (x.status !== "done") {
      const note = x.status === "error" ? esc(x.error) : x.status === "checking" ? "Checking…" : "Waiting…";
      return `<tr class="pending"><td class="fname">${esc(x.name)}<br><small class="muted">${note}</small></td><td></td><td></td><td></td></tr>`;
    }
    const r = x.result;
    const share = r.ai_share;
    return `<tr data-i="${i}" tabindex="0"><td class="fname">${esc(x.name)}</td><td class="num">${r.words.toLocaleString()}</td>
      <td class="num"><span class="share-bar"><i style="width:${Math.round((share ?? 0) * 100)}%"></i></span>${pct(share)}</td>
      <td><span class="v-pill ${r.verdict}">${esc(VERDICTS[r.verdict] || r.verdict)}</span></td></tr>`;
  }).join("");
  els.doc.innerHTML = `<table class="batch-table"><thead><tr><th>File</th><th class="num">Words</th>
    <th class="num">Flagged as AI</th><th>Verdict</th></tr></thead><tbody>${rows}</tbody></table>`;
  els.words.textContent = `${finished} of ${items.length} checked`;
}

function openBatchItem(i) {
  const item = state.batch?.[i];
  if (!item || item.status !== "done") return;
  state.text = item.text;
  state.result = item.result;
  els.input.value = item.text;
  renderReport();
}

function csvCell(v) {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportBatchCsv() {
  const head = ["file", "words", "share_flagged_percent", "document_ai_probability_percent", "verdict",
    "flagged_sentences", "sentences", "hidden_characters", "error"];
  const lines = [head.join(",")];
  for (const x of state.batch || []) {
    const r = x.result;
    const m = r?.manipulation || {};
    lines.push([x.name, r?.words, r?.ai_share == null ? "" : Math.round(r.ai_share * 100),
      r?.document_probability == null ? "" : Math.round(r.document_probability * 100), r ? VERDICTS[r.verdict] : "",
      r ? r.sentences.filter((s) => s.flagged).length : "", r?.sentences.length,
      r ? (m.homoglyphs || 0) + (m.invisible || 0) : "", x.error].map(csvCell).join(","));
  }
  download(`detectai-batch-${new Date().toISOString().slice(0, 10)}.csv`, lines.join("\n"));
}

els.doc.addEventListener("click", (e) => {
  if (state.mode !== "batch") return;
  const row = e.target.closest("tr[data-i]");
  if (row) openBatchItem(Number(row.dataset.i));
});
els.doc.addEventListener("keydown", (e) => {
  if (state.mode === "batch" && e.key === "Enter" && e.target.matches("tr[data-i]")) openBatchItem(Number(e.target.dataset.i));
});
els.batch.addEventListener("click", () => els.files.click());
els.batchAdd.addEventListener("click", () => els.files.click());
els.files.addEventListener("change", () => { startBatch(els.files.files); els.files.value = ""; });
els.batchCsv.addEventListener("click", exportBatchCsv);
els.batchClose.addEventListener("click", () => { state.batch = null; setMode("edit"); syncButtons(); });
els.toBatch.addEventListener("click", () => { setMode("batch"); renderBatch(); });

/* ---------- humanizer ---------- */

const HUMANIZE_STAGES = [
  [0, "Scoring every sentence…"],
  [3, "Rewriting flagged passages, several versions each…"],
  [12, "Checking meaning, names, numbers and tone…"],
  [22, "Re-scoring the best versions with the detector…"],
  [40, "Still working: long texts take a little longer…"],
];

async function humanize() {
  const text = state.text || els.input.value;
  if (!text.trim() || state.busy) return;
  const strength = segValue(els.strength), tone = segValue(els.tone);
  const stages = strength === "light" ? [[0, "Fixing AI-typical wording…"]] : HUMANIZE_STAGES;
  const h = await post("/api/humanize", { text, strength, tone }, stages);
  if (!h) return;
  state.humanized = h;
  state.accepted = new Set(h.changes.map((_, i) => i));
  state.hview = "changes";
  renderHumanized();
}

function composed() {
  const h = state.humanized;
  let out = "", pos = 0;
  h.changes.forEach((c, i) => {
    out += h.original.slice(pos, c.start) + (state.accepted.has(i) ? c.after : c.before);
    pos = c.end;
  });
  return out + h.original.slice(pos);
}

function changesHtml() {
  const h = state.humanized;
  let html = "", pos = 0;
  h.changes.forEach((c, i) => {
    html += esc(h.original.slice(pos, c.start));
    const on = state.accepted.has(i);
    const cls = !on ? "h off" : c.after === "" ? "h del" : `h ${c.kind}`;
    let inner = esc(on && c.after !== "" ? c.after : c.before);
    // "Moreover, t" -> "T": show the dropped opener struck through, then the capitalised letter
    const tail = c.before.slice(-c.after.length);
    if (on && c.kind === "rule" && c.after.length === 1 && tail.toLowerCase() === c.after.toLowerCase()) {
      inner = `<s>${esc(c.before.slice(0, -1))}</s>${esc(c.after)}`;
    }
    html += `<mark class="${cls}" data-k="h" data-i="${i}">${inner}</mark>`;
    pos = c.end;
  });
  return html + esc(h.original.slice(pos));
}

function renderHumanized() {
  const h = state.humanized;
  for (const b of els.htabs.querySelectorAll("button")) b.setAttribute("aria-selected", String(b.dataset.hview === state.hview));
  if (state.hview === "split") {
    els.doc.innerHTML = `<div class="split"><div><h3>Original</h3><div>${esc(h.original)}</div></div>
      <div><h3>Humanized</h3><div>${esc(composed())}</div></div></div>`;
  } else if (state.hview === "clean") {
    els.doc.textContent = composed();
  } else {
    els.doc.innerHTML = changesHtml();
  }

  els.hBefore.textContent = pct(h.before.ai_share);
  els.hAfter.textContent = pct(h.after.ai_share);
  els.hBeforeV.textContent = VERDICTS[h.before.verdict] || "";
  els.hAfterV.textContent = VERDICTS[h.after.verdict] || "";
  const rewrites = h.changes.filter((c) => c.kind === "rewrite").length;
  const facts = [
    ["Passages rewritten", rewrites],
    ["Word & phrase fixes", h.changes.length - rewrites],
    ["Changes kept", `${state.accepted.size} of ${h.changes.length}`],
    ["Meaning kept", rewrites ? pct(h.similarity) : "–"],
    ["Document AI probability", `${pct(h.before.document_probability)} → ${pct(h.after.document_probability)}`],
    ["Rewriter", h.backend === "rules" ? "Rules only" : h.backend.replace(/^local:/, "").replace(/^claude:/, "Claude ")],
    ["Time", `${(h.elapsed_ms / 1000).toFixed(1)}s`],
  ];
  els.hFacts.innerHTML = facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("");
  els.words.textContent = `${countWords(composed()).toLocaleString()} words`;
  if (state.mode !== "humanized") setMode("humanized");
}

function toggleChange(i) {
  if (state.accepted.has(i)) state.accepted.delete(i); else state.accepted.add(i);
  renderHumanized();
}

els.htabs.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-hview]");
  if (b && state.humanized) { state.hview = b.dataset.hview; renderHumanized(); }
});

/* ---------- tooltips ---------- */

function tipHtml(k, i) {
  if (k === "h") {
    const c = state.humanized.changes[i];
    const on = state.accepted.has(i);
    const scores = c.kind === "rewrite"
      ? `<br>AI likelihood ${pct(c.ai_before)} → ${pct(c.ai_after)} · meaning kept ${pct(c.similarity)}` : "";
    const other = on ? `<b>Original:</b> ${esc(c.before)}` : `<b>Suggested:</b> ${esc(c.after || "(remove)")}`;
    return `${other}${scores}<ul>${c.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <span>Click to ${on ? "keep the original" : "use the suggestion"}.</span>`;
  }
  const r = state.result;
  if (k === "s") {
    const s = r.sentences[i];
    const head = s.probability == null ? "Not scored" : `<b>${pct(s.probability)}</b> AI likelihood${s.flagged ? " · flagged" : ""}`;
    return `${head}<br><span>Click for details and rewrite suggestions.</span>`;
  }
  if (k === "w") {
    const t = r.tokens[i];
    const word = state.text.slice(t.s, t.e);
    const rows = t.alts.map(([alt, p]) => {
      const hit = alt.trim() === word.trim();
      const shown = alt.includes("\n") ? "↵ (new line)" : alt.trim() || "␣ (space)";
      return `<span class="${hit ? "hit" : ""}">${esc(shown)}</span><span class="${hit ? "hit" : ""}">${(p * 100).toFixed(1)}%</span>`;
    }).join("");
    const rank = t.r === 1 ? "the model's top guess" : `the model's #${t.r.toLocaleString()} guess`;
    return `<b>${esc(word)}</b> was ${rank} (${(Math.exp(t.lp) * 100).toFixed(1)}%)<div class="alts">${rows}</div>`;
  }
  const p = r.patterns[i];
  return `<b>${esc(p.label)}</b><br>${esc(p.why)}`;
}

function showTip(target, x, y) {
  if (!els.pop.hidden) return;
  els.tip.innerHTML = tipHtml(target.dataset.k, Number(target.dataset.i));
  show(els.tip, true);
  const { offsetWidth: w, offsetHeight: h } = els.tip;
  const left = Math.min(x + 14, window.innerWidth - w - 8);
  const top = y + 18 + h > window.innerHeight ? y - h - 12 : y + 18;
  els.tip.style.left = `${Math.max(8, left)}px`;
  els.tip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() { show(els.tip, false); }

els.doc.addEventListener("mousemove", (e) => {
  const m = e.target.closest("mark");
  if (m) showTip(m, e.clientX, e.clientY); else hideTip();
});
els.doc.addEventListener("mouseleave", hideTip);
els.doc.addEventListener("click", (e) => {
  const m = e.target.closest("mark");
  if (!m) return;
  if (m.dataset.k === "h") {
    toggleChange(Number(m.dataset.i));
    const again = els.doc.querySelector(`mark[data-k="h"][data-i="${m.dataset.i}"]`);
    if (again) showTip(again, e.clientX, e.clientY);
  } else if (m.dataset.k === "s") {
    openPop(Number(m.dataset.i), e.clientX, e.clientY);
  } else {
    showTip(m, e.clientX, e.clientY);  // touch screens have no hover
  }
});
document.addEventListener("scroll", () => { hideTip(); closePop(); }, true);

/* ---------- accuracy panel ---------- */

let accuracyData = null;

async function openAccuracy() {
  els.accuracy.showModal();
  try {
    accuracyData = accuracyData || await (await fetch("/accuracy.json", { cache: "no-store" })).json();
    renderAccuracy(state.sensitivity);
  } catch {
    els.accuracyBody.innerHTML = `<p class="muted">No evaluation results yet. Run <code>python -m ml.eval.evaluate --compare</code>.</p>`;
  }
}

function renderAccuracy(level) {
  const a = accuracyData;
  const levels = a.suites[0]?.levels ? Object.keys(a.suites[0].levels) : ["strict"];
  if (!levels.includes(level)) level = "strict";
  const comp = new Map((a.comparison?.suites || []).map((s) => [s.name, s]));
  const flagCls = (x) => (x <= 0.01 ? "acc-good" : x >= 0.1 ? "acc-bad" : "");
  const rows = a.suites.map((s) => {
    const ours = s.levels?.[level] || s;
    const c = comp.get(s.name);
    return `<tr><td>${esc(s.name)}<small>${esc(s.detail || "")} · ${s.n_human} human, ${s.n_ai} AI texts</small></td>
      <td class="${flagCls(ours.human_flagged)}">${pct(ours.human_flagged)}</td><td>${pct(ours.ai_caught)}</td>
      ${a.comparison ? `<td class="${c ? flagCls(c.human_flagged) : ""}">${c ? pct(c.human_flagged) : "–"}</td><td>${c ? pct(c.ai_caught) : "–"}</td>` : ""}</tr>`;
  }).join("");
  const picker = levels.length > 1 ? `<div class="field"><span class="field-label">Numbers for sensitivity</span>
      <div class="seg" id="acc-level" role="radiogroup" aria-label="Sensitivity">${levels.map((l) =>
        `<button role="radio" data-v="${l}" aria-checked="${l === level}">${l[0].toUpperCase()}${l.slice(1)}</button>`).join("")}</div></div>` : "";
  els.accuracyBody.innerHTML = `
    <p>Every number below comes from texts the detector never saw while it was being trained, last measured on
      ${esc(a.date)}. Like Turnitin, DetectAI is tuned to <b>avoid accusing people</b>: by default it would rather miss
      some AI text than flag a human.</p>
    ${picker}
    <table class="acc-table">
      <thead><tr><th>Test set</th><th>Humans wrongly flagged</th><th>AI caught</th>
        ${a.comparison ? `<th>Public detector:<br>humans wrongly flagged</th><th>Public detector:<br>AI caught</th>` : ""}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${a.comparison ? `<p class="muted small">Public detector: ${esc(a.comparison.model)}, a RoBERTa model trained on the RAID benchmark,
      run on the same texts at its default cut-off. High detection rates mean little if real students are flagged too.</p>` : ""}
    <h3>What to keep in mind</h3>
    <ul>
      <li>Writing from the newest AI models (2026) is the hardest to catch; at the Strict setting expect many of those texts to pass.</li>
      <li>Short texts (under 150 words) get no verdict. Lists, formulas, famous passages and heavily edited text are unreliable.</li>
      <li>A score is a reason to look closer, never proof. Ask for drafts or notes before drawing conclusions.</li>
    </ul>
    <p class="muted small">Detector setup: ${esc(a.model)}. Full reports: <code>data/reports/</code> in the project.</p>`;
}

els.accuracyBody.addEventListener("click", (e) => {
  const b = e.target.closest("#acc-level button[data-v]");
  if (b && accuracyData) renderAccuracy(b.dataset.v);
});
$("open-accuracy").addEventListener("click", openAccuracy);
$("close-accuracy").addEventListener("click", () => els.accuracy.close());
els.accuracy.addEventListener("click", (e) => { if (e.target === els.accuracy) els.accuracy.close(); });

/* ---------- files, printing & wiring ---------- */

async function loadFile(file) {
  if (!file) return;
  show(els.error, false);
  try {
    els.input.value = await readTextFile(file);
    syncButtons();
    els.input.focus();
  } catch (err) {
    els.error.textContent = err.message || String(err);
    show(els.error, true);
  }
}

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

for (const group of [els.strength, els.tone]) {
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-v]");
    if (!b) return;
    for (const x of group.querySelectorAll("button")) x.setAttribute("aria-checked", String(x === b));
    syncButtons();
  });
}

els.input.addEventListener("input", () => { syncButtons(); scheduleLive(); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
});
els.analyze.addEventListener("click", () => analyze());
els.edit.addEventListener("click", () => { setMode("edit"); syncButtons(); els.input.focus(); });
els.print.addEventListener("click", () => {
  if (state.view !== "sentences") setView("sentences");
  window.print();
});
els.toHumanize.addEventListener("click", () => els.humanize.scrollIntoView({ behavior: "smooth", block: "center" }));
els.humanize.addEventListener("click", humanize);
els.backReport.addEventListener("click", () => { setMode("report"); setView(state.view); });
els.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(composed());
    els.copy.textContent = "Copied ✓";
  } catch {
    els.copy.textContent = "Copy failed: use Download";
  }
  setTimeout(() => { els.copy.textContent = "Copy text"; }, 1800);
});
els.download.addEventListener("click", () => download("humanized.txt", composed()));
els.again.addEventListener("click", () => {  // second pass over the current result, with your choices applied
  const t = composed();
  state.text = t;
  els.input.value = t;
  humanize();
});
els.recheck.addEventListener("click", () => { const t = composed(); els.input.value = t; analyze(t); });
document.querySelector(".samples").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-sample]");
  if (!b) return;
  els.input.value = SAMPLES[b.dataset.sample];
  syncButtons();
  analyze();
});
els.upload.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => { loadFile(els.file.files[0]); els.file.value = ""; });
els.input.addEventListener("dragover", (e) => e.preventDefault());
els.input.addEventListener("drop", (e) => {
  if (!e.dataTransfer.files.length) return;  // plain dragged text falls through to the textarea
  e.preventDefault();
  if (e.dataTransfer.files.length > 1) startBatch(e.dataTransfer.files);  // several essays: check them all
  else loadFile(e.dataTransfer.files[0]);
});
els.tabs.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]");
  if (b && state.result) setView(b.dataset.view);
});

// #demo, #demo=words, #demo=humanize, #demo=humanize-light, #demo=pop, #demo=accuracy: quick checks and screenshots.
async function runDemoFromHash() {
  const m = location.hash.match(/^#demo(?:=([\w-]+))?$/);
  if (!m) return;
  const arg = m[1] || "";
  if (arg === "accuracy") { openAccuracy(); return; }
  if (arg === "batch") {
    startBatch(Object.entries(SAMPLES).map(([k, t]) => new File([t], `${k}-essay.txt`, { type: "text/plain" })));
    return;
  }
  if (VIEWS[arg]) state.view = arg;
  els.input.value = SAMPLES[arg === "human" || arg === "mixed" ? arg : "ai"];
  syncButtons();
  await analyze();
  if (arg === "pop") {
    const mark = els.doc.querySelector('mark[data-k="s"]');
    const box = mark.getBoundingClientRect();
    openPop(0, box.left + 120, box.bottom);
    await suggest();
  }
  if (arg.startsWith("humanize")) {
    if (arg === "humanize-light") els.strength.querySelector('[data-v="light"]').click();
    await humanize();
  }
}

setMode("edit");
syncButtons();
pollHealth();
