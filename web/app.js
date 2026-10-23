"use strict";

const SAMPLE = `In today's fast-paced digital world, social media plays a pivotal role in shaping how students learn, communicate, and build relationships. Platforms like Instagram, TikTok, and X have transformed the educational landscape, offering both remarkable opportunities and significant challenges.

On one hand, social media fosters collaboration and provides access to a wide range of educational resources. Students can join study groups, follow experts in their field, and engage with diverse perspectives from around the globe. Moreover, these platforms can serve as a powerful tool for creativity, allowing learners to showcase their work and receive valuable feedback.

On the other hand, it is important to note that excessive use can have a profound impact on academic performance and mental well-being. Constant notifications disrupt focus, while the pressure to maintain a curated online presence can lead to anxiety and low self-esteem. Furthermore, the spread of misinformation underscores the need for strong digital literacy skills.

Ultimately, the key lies in striking a balance. By setting healthy boundaries, practicing mindful consumption, and leveraging social media as a learning tool rather than a distraction, students can harness its benefits while minimizing its drawbacks. Schools and parents also play a crucial role in guiding young people as they navigate the complexities of the digital age. In conclusion, social media is not inherently good or bad — it is a tool whose impact depends on how we choose to use it.`;

const VERDICTS = {
  likely_ai: "Likely AI-generated",
  mixed: "Mix of AI and human writing",
  likely_human: "Likely human-written",
  too_short: "Too short to judge",
  uncalibrated: "Model not calibrated",
};

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), statusText: $("status-text"), input: $("input"), doc: $("doc"), tabs: $("tabs"),
  title: $("pane-title"), words: $("word-count"), sample: $("sample"), edit: $("edit"), analyze: $("analyze"),
  upload: $("upload"), file: $("file"), toHumanize: $("to-humanize"), humanize: $("humanize"),
  strength: $("strength"), tone: $("tone"), humanizeNote: $("humanize-note"),
  backReport: $("back-report"), download: $("download"), recheck: $("recheck"), copy: $("copy"),
  empty: $("empty"), loading: $("loading"), loadingText: $("loading-text"), error: $("error"),
  results: $("results"), hresults: $("hresults"), ringFill: $("ring-fill"),
  share: $("share"), shareCaption: $("share-caption"), verdict: $("verdict"), verdictDetail: $("verdict-detail"),
  signals: $("signals"), legend: $("legend"), tip: $("tip"),
  hBefore: $("h-before"), hAfter: $("h-after"), hBeforeV: $("h-before-v"), hAfterV: $("h-after-v"), hFacts: $("h-facts"),
};

const state = {
  ready: false, busy: false, humanizer: "waiting", mode: "edit",
  result: null, text: "", view: "sentences",
  humanized: null, accepted: new Set(),
};

const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const countWords = (t) => (t.match(/[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*/g) || []).length;
const show = (el, on) => { el.hidden = !on; };
const segValue = (group) => group.querySelector('[aria-checked="true"]').dataset.v;

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
      const rw = h.humanizer === "ready" ? " · humanizer ready" : h.humanizer === "loading" ? " · humanizer loading…" : "";
      els.statusText.textContent = `${h.model.split("+")[0]} · ${h.device.toUpperCase()}${h.calibrated ? "" : " · uncalibrated"}${rw}`;
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
    els.statusText.textContent = "Server not reachable";
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
  show(els.results, mode === "report");
  show(els.hresults, mode === "humanized");
  show(els.empty, mode === "edit" && !state.result);
  els.title.textContent = { edit: "Check a text for AI writing", report: "Report", humanized: "Humanized text" }[mode];
  hideTip();
}

async function post(url, body, loadingText) {
  state.busy = true;
  syncButtons();
  show(els.error, false);
  els.loadingText.textContent = loadingText;
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
    state.busy = false;
    show(els.loading, false);
    syncButtons();
  }
}

/* ---------- detection report ---------- */

async function analyze(text = els.input.value) {
  if (!text.trim() || !state.ready || state.busy) return;
  show(els.empty, false);
  const r = await post("/api/detect", { text }, "Reading your text word by word…");
  if (!r) return;
  state.result = r;
  state.text = text;
  renderReport();
}

function renderReport() {
  const r = state.result;
  const share = r.ai_share;
  els.share.textContent = pct(share);
  els.shareCaption.textContent = share == null ? "not scored" : "flagged as AI";
  els.ringFill.style.strokeDashoffset = String(326.7 * (1 - (share ?? 0)));

  els.verdict.textContent = VERDICTS[r.verdict] || r.verdict;
  els.verdict.dataset.v = r.verdict;
  const flagged = r.sentences.filter((s) => s.flagged).length;
  els.verdictDetail.textContent = r.too_short
    ? `Add more text. At least ${r.min_words} words are needed for a reliable score; this has ${r.words}. The word view still works.`
    : `${flagged} of ${r.sentences.length} sentences flagged. Whole-document AI probability: ${pct(r.document_probability)}.`;

  els.signals.innerHTML = r.signals.map((s) => {
    const c = Math.max(-1, Math.min(1, s.contribution / 3));
    const w = Math.abs(c) * 50;
    const side = c >= 0 ? `left:50%;width:${w}%;background:rgb(var(--ai))` : `left:${50 - w}%;width:${w}%;background:rgb(var(--human))`;
    return `<li class="signal">
      <div class="signal-top"><b>${esc(s.label)}</b><span>${esc(s.value)}</span></div>
      <div class="bar" title="${c >= 0 ? "Leans AI" : "Leans human"}"><i style="${side}"></i></div>
      <p>${esc(s.description)}</p></li>`;
  }).join("");

  const m = r.manipulation || { homoglyphs: 0, invisible: 0 };
  const tricks = [];
  if (m.homoglyphs) tricks.push(`${m.homoglyphs} look-alike letter${m.homoglyphs === 1 ? "" : "s"} from other alphabets`);
  if (m.invisible) tricks.push(`${m.invisible} invisible character${m.invisible === 1 ? "" : "s"}`);
  $("manipulation").innerHTML = tricks.length
    ? `<b>Hidden characters found:</b> ${esc(tricks.join(" and "))}. These are a common trick to fool detectors and plagiarism checkers; the text was analysed with them undone.`
    : "";
  show($("manipulation"), tricks.length > 0);

  els.words.textContent = `${r.words.toLocaleString()} words · checked in ${(r.elapsed_ms / 1000).toFixed(1)}s`;
  setMode("report");
  setView(state.view);
}

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
    <p class="muted small" style="margin:4px 0 0">Each sentence is judged together with its neighbours. Hover one to see why.</p>`,
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
}

/* ---------- humanizer ---------- */

async function humanize() {
  const text = state.text || els.input.value;
  if (!text.trim() || state.busy) return;
  const strength = segValue(els.strength), tone = segValue(els.tone);
  const msg = strength === "light" ? "Fixing AI-typical wording…" : "Rewriting flagged passages and checking each version… this can take a minute.";
  const h = await post("/api/humanize", { text, strength, tone }, msg);
  if (!h) return;
  state.humanized = h;
  state.accepted = new Set(h.changes.map((_, i) => i));
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

function renderHumanized() {
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
  els.doc.innerHTML = html + esc(h.original.slice(pos));

  els.hBefore.textContent = pct(h.before.ai_share);
  els.hAfter.textContent = pct(h.after.ai_share);
  els.hBeforeV.textContent = VERDICTS[h.before.verdict] || "";
  els.hAfterV.textContent = VERDICTS[h.after.verdict] || "";
  const rewrites = h.changes.filter((c) => c.kind === "rewrite").length;
  const facts = [
    ["Passages rewritten", rewrites],
    ["Word & phrase fixes", h.changes.length - rewrites],
    ["Meaning kept", rewrites ? pct(h.similarity) : "–"],
    ["Document AI probability", `${pct(h.before.document_probability)} → ${pct(h.after.document_probability)}`],
    ["Rewriter", h.backend === "rules" ? "Rules only" : h.backend.replace(/^local:/, "").replace(/^claude:/, "Claude ")],
    ["Time", `${(h.elapsed_ms / 1000).toFixed(1)}s`],
  ];
  els.hFacts.innerHTML = facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("");
  els.words.textContent = `${countWords(composed()).toLocaleString()} words`;
  setMode("humanized");
}

function toggleChange(i) {
  if (state.accepted.has(i)) state.accepted.delete(i); else state.accepted.add(i);
  renderHumanized();
}

/* ---------- tooltips ---------- */

function tipHtml(k, i) {
  if (k === "h") {
    const c = state.humanized.changes[i];
    const on = state.accepted.has(i);
    const scores = c.kind === "rewrite"
      ? `<br>AI likelihood ${pct(c.ai_before)} → ${pct(c.ai_after)} · meaning kept ${pct(c.similarity)}` : "";
    const other = on ? `<b>Original:</b> ${esc(c.before)}` : `<b>Suggested:</b> ${esc(c.after || "(remove)")}`;
    return `${other}${scores}<ul>${c.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <span class="muted">Click to ${on ? "keep the original" : "use the suggestion"}.</span>`;
  }
  const r = state.result;
  if (k === "s") {
    const s = r.sentences[i];
    const head = s.probability == null ? "Not scored" : `<b>${pct(s.probability)}</b> AI likelihood${s.flagged ? " · flagged" : ""}`;
    const note = r.too_short ? `<br><span>Text too short for sentence flags.</span>` : "";
    const reasons = s.reasons.length ? `<ul>${s.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
    return head + note + reasons;
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
  } else {
    showTip(m, e.clientX, e.clientY);  // touch screens have no hover
  }
});
document.addEventListener("scroll", hideTip, true);

/* ---------- files & wiring ---------- */

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

els.input.addEventListener("input", syncButtons);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
});
els.analyze.addEventListener("click", () => analyze());
els.edit.addEventListener("click", () => { setMode("edit"); syncButtons(); els.input.focus(); });
els.toHumanize.addEventListener("click", () => els.humanize.scrollIntoView({ behavior: "smooth", block: "center" }));
els.humanize.addEventListener("click", humanize);
els.backReport.addEventListener("click", () => { setMode("report"); setView(state.view); });
els.copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(composed());
  els.copy.textContent = "Copied ✓";
  setTimeout(() => { els.copy.textContent = "Copy text"; }, 1500);
});
els.download.addEventListener("click", () => download("humanized.txt", composed()));
els.recheck.addEventListener("click", () => { const t = composed(); els.input.value = t; analyze(t); });
els.sample.addEventListener("click", () => { els.input.value = SAMPLE; syncButtons(); els.input.focus(); });
els.upload.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => { loadFile(els.file.files[0]); els.file.value = ""; });
els.input.addEventListener("dragover", (e) => e.preventDefault());
els.input.addEventListener("drop", (e) => {
  if (!e.dataTransfer.files.length) return;  // plain dragged text falls through to the textarea
  e.preventDefault();
  loadFile(e.dataTransfer.files[0]);
});
els.tabs.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]");
  if (b && state.result) setView(b.dataset.view);
});

// #demo, #demo=words, #demo=humanize, #demo=humanize-light: load the sample and run it (quick checks, screenshots).
async function runDemoFromHash() {
  const m = location.hash.match(/^#demo(?:=([\w-]+))?$/);
  if (!m) return;
  const arg = m[1] || "";
  if (VIEWS[arg]) state.view = arg;
  els.input.value = SAMPLE;
  syncButtons();
  await analyze();
  if (arg.startsWith("humanize")) {
    if (arg === "humanize-light") els.strength.querySelector('[data-v="light"]').click();
    await humanize();
  }
}

setMode("edit");
syncButtons();
pollHealth();
