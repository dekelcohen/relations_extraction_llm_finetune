/* Small vanilla-JS dataset viewer.
   Served by viewer/server.py (FastAPI): fetches /api/datasets + /api/dataset/<name>, which read the
   source files on every request (single source of truth -- no exported/copied data). */

const RECORD_HUES = [217, 142, 32, 4, 262, 190, 330, 83]; // blue, green, amber, red, violet, cyan, pink, lime
const VERDICT_SYMBOL = { ok: "\u2713", wrong: "\u2717", missing: "\u2205", hit: "\u26a0" };

// `var` (not `const`/`let`) for `DATASET`, `state` and `MODEL_CODES`: these need to be real properties
// of `window`/`globalThis` so they're reachable when app.js is `eval`'d from a test harness (see smoke_test.js).
var DATASET = { samples: [] };
var state = {
  query: "",
  langs: new Set(),
  formats: new Set(),
  tiers: new Set(),
  hardCases: new Set(),
  flags: new Set(),
  activeModels: new Set(), // which model(s)' predictions are overlaid on the document text
  selectedId: null,
  view: "doc", // "doc" | "prompts"
};

var MODEL_CODES = {};

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
}

function norm(s) {
  return (s || "").toString().trim().toLowerCase().replace(/[\s\u200f\u200e"'`\u05f3\u05f4.,]/g, "");
}

// ---------- dataset loading (from viewer/server.py) ----------

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function loadDataset(name) {
  DATASET = await fetchJson(`api/dataset/${encodeURIComponent(name)}`);
  afterDatasetLoaded();
}

function afterDatasetLoaded() {
  MODEL_CODES = assignModelCodes(allModels());
  state.activeModels = new Set();
  state.selectedId = null;
  initFilters();
  renderModelToggles();
  renderList();
  document.getElementById("sample-view").hidden = true;
  document.getElementById("empty-state").hidden = DATASET.samples.length !== 0;
  document.getElementById("empty-state").textContent = "Select a sample on the left.";
  const hashId = decodeURIComponent((window.location.hash || "").slice(1));
  if (hashId && DATASET.samples.some((s) => s.id === hashId)) selectSample(hashId);
  else if (DATASET.samples.length) selectSample(DATASET.samples[0].id);
}

async function initDatasetSelector() {
  const select = document.getElementById("dataset-select");
  try {
    const datasets = await fetchJson("api/datasets");
    if (!datasets.length) throw new Error("no datasets found on disk");
    select.innerHTML = "";
    for (const d of datasets) select.appendChild(el("option", { value: d.name, text: d.label }, []));
    const dflt = datasets.find((d) => d.default) || datasets[0];
    select.value = dflt.name;
    select.addEventListener("change", () => loadDataset(select.value));
    await loadDataset(dflt.name);
  } catch (e) {
    console.error("dataset API unavailable:", e);
    const empty = document.getElementById("empty-state");
    empty.hidden = false;
    empty.textContent = `Could not load data from the viewer server (${e.message}). Run: python viewer/server.py --data-root data`;
  }
}

// ---------- filters & sidebar ----------

function uniqueSorted(values) {
  return [...new Set(values)].filter((v) => v != null).sort();
}

function buildChipGroup(containerId, values, selectedSet) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  for (const v of values) {
    const chip = el("span", {
      class: "chip" + (selectedSet.has(v) ? " active" : ""),
      text: v,
      onclick: () => {
        selectedSet.has(v) ? selectedSet.delete(v) : selectedSet.add(v);
        chip.classList.toggle("active");
        renderList();
      },
    });
    container.appendChild(chip);
  }
}

function initFilters() {
  state.langs.clear();
  state.formats.clear();
  state.tiers.clear();
  state.hardCases.clear();
  state.flags.clear();
  buildChipGroup("lang-filters", uniqueSorted(DATASET.samples.map((s) => s.lang)), state.langs);
  buildChipGroup("format-filters", uniqueSorted(DATASET.samples.map((s) => s.format)), state.formats);
  buildChipGroup("tier-filters", uniqueSorted(DATASET.samples.map((s) => s.tier)), state.tiers);
  buildChipGroup(
    "hardcase-filters",
    uniqueSorted(DATASET.samples.flatMap((s) => s.hard_cases)),
    state.hardCases
  );
  buildChipGroup("flag-filters", uniqueSorted(DATASET.samples.flatMap((s) => s.flags || [])), state.flags);
}

document.getElementById("search").addEventListener("input", (e) => {
  state.query = e.target.value.trim().toLowerCase();
  renderList();
});

function sampleMatches(s) {
  if (state.langs.size && !state.langs.has(s.lang)) return false;
  if (state.formats.size && !state.formats.has(s.format)) return false;
  if (state.tiers.size && !state.tiers.has(s.tier)) return false;
  if (state.hardCases.size && ![...state.hardCases].every((hc) => s.hard_cases.includes(hc))) return false;
  if (state.flags.size && ![...state.flags].some((f) => (s.flags || []).includes(f))) return false;
  if (state.query) {
    const hay = `${s.id} ${s.genre || ""} ${s.schema_id}`.toLowerCase();
    if (!hay.includes(state.query)) return false;
  }
  return true;
}

function renderList() {
  const list = document.getElementById("sample-list");
  list.innerHTML = "";
  const filtered = DATASET.samples.filter(sampleMatches);
  document.getElementById("sample-count").textContent = `${filtered.length} / ${DATASET.samples.length} samples`;
  for (const s of filtered) {
    const li = el(
      "li",
      { onclick: () => selectSample(s.id), class: s.id === state.selectedId ? "selected" : "" },
      [
        el("span", { class: "sid", text: `${s.id}` }),
        s.tier ? el("span", { class: `tier-tag tier-${s.tier}`, text: s.tier }) : null,
        (s.flags || []).length ? el("span", { class: "flag-tag", title: s.flags.join(", "), text: "\u2605" }) : null,
      ]
    );
    list.appendChild(li);
  }
}

// ---------- per-model prediction overlay (errors on top of GT) ----------

function allModels() {
  return uniqueSorted(DATASET.samples.flatMap((s) => Object.keys(s.preds || {})));
}

function assignModelCodes(names) {
  const codes = {};
  const used = new Set();
  for (const name of names) {
    let len = 1;
    let code = name.slice(0, len).toUpperCase();
    while (used.has(code) && len < name.length) {
      len += 1;
      code = name.slice(0, len).toUpperCase();
    }
    while (used.has(code)) code += "'"; // extremely unlikely fallback for exact-duplicate prefixes
    used.add(code);
    codes[name] = code;
  }
  return codes;
}

function renderModelToggles() {
  const container = document.getElementById("model-toggles");
  container.innerHTML = "";
  const models = allModels();
  if (!models.length) return;
  container.appendChild(el("span", { class: "muted", text: "Predictions:" }));
  for (const model of models) {
    const chip = el("span", {
      class: "chip" + (state.activeModels.has(model) ? " active" : ""),
      text: `${model} (${MODEL_CODES[model]})`,
      title: `Toggle ${model} predictions overlaid on the document as \u2713/\u2717/\u2205 badges`,
      onclick: () => {
        state.activeModels.has(model) ? state.activeModels.delete(model) : state.activeModels.add(model);
        chip.classList.toggle("active");
        refreshCurrent();
      },
    });
    container.appendChild(chip);
  }
}

/** Gold-slot verdict for one model: {verdict, predValue?} | null (nothing to say, or no gold value). */
/** Scorer verdict (export-time: same record alignment + partial-span matching as slot_f1), if available. */
function scorerVerdict(sample, model, recordIdx, slot) {
  const vs = sample.preds[model] && sample.preds[model].verdicts;
  return vs && vs[recordIdx] ? vs[recordIdx][slot] : undefined;
}

function slotVerdict(sample, model, recordIdx, slot, gold) {
  const predRecords = sample.preds[model] && sample.preds[model].records;
  if (!predRecords || !gold) return null;
  const sv = scorerVerdict(sample, model, recordIdx, slot);
  if (sv) return sv.v === "ok" || sv.v === "wrong" ? { verdict: sv.v, predValue: sv.pred } : { verdict: "missing" };
  const predRec = predRecords[recordIdx];
  const pv = predRec ? predRec[slot] : undefined;
  if (pv === undefined || pv === null || pv === "") return { verdict: "missing" };
  return { verdict: norm(pv) === norm(gold) ? "ok" : "wrong", predValue: pv };
}

/** Did a model's prediction match a planted *distractor* instead of the gold value (decoy-hit)? */
function decoyVerdict(sample, model, decoy) {
  const predRecords = sample.preds[model] && sample.preds[model].records;
  const predRec = predRecords && predRecords[decoy.record];
  const sv = scorerVerdict(sample, model, decoy.record, decoy.slot);
  const pv = sv ? sv.pred : predRec ? predRec[decoy.slot] : undefined;
  if (pv && norm(pv) === norm(decoy.value)) return { verdict: "hit", predValue: pv };
  return null;
}

function buildBadges(span, sample) {
  if (!state.activeModels.size) return [];
  const badges = [];
  for (const model of state.activeModels) {
    let res = null;
    if (span.cls === "ent-decoy") res = decoyVerdict(sample, model, span.decoy);
    else if (span.cls === "ent-main") res = slotVerdict(sample, model, span.recordIdx, "main_entity", span.value);
    else res = slotVerdict(sample, model, span.recordIdx, span.slot, span.value);
    if (!res) continue;
    const code = MODEL_CODES[model] || model[0];
    const title = `${model}: ${res.verdict}` + (res.predValue ? ` (predicted "${res.predValue}")` : "");
    badges.push(el("sup", { class: `verdict verdict-${res.verdict}`, title, text: `${VERDICT_SYMBOL[res.verdict]}${code}` }));
  }
  return badges;
}

// ---------- span computation & highlighting ----------

function schemaSlotType(schemaId, slotName) {
  const schema = DATASET.schemas[schemaId];
  const slot = schema && schema.slots.find((sl) => sl.name === slotName);
  return slot ? slot.type : "text";
}

function buildSpans(sample, showDecoys) {
  const spans = [];
  (sample.records || []).forEach((rec, ri) => {
    const hue = RECORD_HUES[ri % RECORD_HUES.length];
    if (rec.main_entity) {
      spans.push({ value: rec.main_entity, cls: "ent-main", hue, recordIdx: ri, slot: null,
        title: `main entity (record ${ri + 1})` });
    }
    for (const [slot, value] of Object.entries(rec)) {
      if (slot === "main_entity" || value == null || value === "") continue;
      const type = schemaSlotType(sample.schema_id, slot);
      spans.push({ value, cls: "ent-slot", hue, recordIdx: ri, slot,
        title: `${slot} (${type}) \u2014 record ${ri + 1}` });
    }
  });
  if (showDecoys) {
    for (const d of sample.decoys || []) {
      if (!d.value) continue;
      spans.push({ value: d.value, cls: "ent-decoy", hue: 0, recordIdx: d.record, slot: d.slot, decoy: d,
        title: `distractor (${d.type}): ${d.role}` });
    }
  }
  // Longer values first, so e.g. a full organization name is matched before a shorter value
  // that happens to be its substring.
  return spans.sort((a, b) => b.value.length - a.value.length);
}

function findMatches(text, spans) {
  const claimed = []; // [start, end)
  const overlaps = (s, e) => claimed.some(([cs, ce]) => s < ce && e > cs);
  const matches = [];
  for (const span of spans) {
    if (!span.value) continue;
    let from = 0;
    while (true) {
      const idx = text.indexOf(span.value, from);
      if (idx === -1) break;
      const end = idx + span.value.length;
      if (!overlaps(idx, end)) {
        matches.push({ start: idx, end, span });
        claimed.push([idx, end]);
      }
      from = end;
    }
  }
  matches.sort((a, b) => a.start - b.start);
  return matches;
}

function markFor(span) {
  const mark = el("mark", { class: `ent-${span.cls === "ent-main" ? "main" : span.cls === "ent-slot" ? "slot" : "decoy"}`, title: span.title });
  mark.style.setProperty("--hue", span.hue);
  return mark;
}

function appendMatch(target, m, text, sample) {
  const mark = markFor(m.span);
  mark.textContent = text.slice(m.start, m.end);
  target.appendChild(mark);
  for (const badge of buildBadges(m.span, sample)) target.appendChild(badge);
}

function highlightPlainText(container, text, spans, sample) {
  container.textContent = "";
  const matches = findMatches(text, spans);
  let cursor = 0;
  for (const m of matches) {
    if (m.start > cursor) container.appendChild(document.createTextNode(text.slice(cursor, m.start)));
    appendMatch(container, m, text, sample);
    cursor = m.end;
  }
  if (cursor < text.length) container.appendChild(document.createTextNode(text.slice(cursor)));
}

function highlightHtmlNode(root, spans, sample) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  for (const node of textNodes) {
    const text = node.nodeValue;
    const matches = findMatches(text, spans);
    if (!matches.length) continue;
    const frag = document.createDocumentFragment();
    let cursor = 0;
    for (const m of matches) {
      if (m.start > cursor) frag.appendChild(document.createTextNode(text.slice(cursor, m.start)));
      appendMatch(frag, m, text, sample);
      cursor = m.end;
    }
    if (cursor < text.length) frag.appendChild(document.createTextNode(text.slice(cursor)));
    node.parentNode.replaceChild(frag, node);
  }
}

function renderDoc(sample) {
  const container = document.getElementById("doc-container");
  container.innerHTML = "";
  container.setAttribute("dir", sample.lang === "en" ? "ltr" : "rtl");
  const showDecoys = document.getElementById("toggle-decoys").checked;
  const spans = buildSpans(sample, showDecoys);

  if (sample.format === "html") {
    const doc = new DOMParser().parseFromString(sample.text, "text/html");
    highlightHtmlNode(doc.body, spans, sample);
    container.dir = doc.documentElement.getAttribute("dir") || container.dir;
    while (doc.body.firstChild) container.appendChild(doc.body.firstChild);
  } else {
    highlightPlainText(container, sample.text, spans, sample);
  }
}

function renderLegend(sample) {
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  (sample.records || []).forEach((rec, ri) => {
    const hue = RECORD_HUES[ri % RECORD_HUES.length];
    const swatch = el("span", { class: "legend-swatch" });
    swatch.style.background = `hsl(${hue}, 70%, 55%)`;
    legend.appendChild(el("span", { class: "legend-item" }, [swatch, rec.main_entity || `record ${ri + 1}`]));
  });
  if (document.getElementById("toggle-decoys").checked && sample.decoys && sample.decoys.length) {
    const swatch = el("span", { class: "legend-swatch" });
    swatch.style.background = "#9a9aa6";
    legend.appendChild(el("span", { class: "legend-item" }, [swatch, "distractor (dashed)"]));
  }
  if (state.activeModels.size) {
    legend.appendChild(el("span", { class: "legend-item",
      text: "\u2713 correct  \u2717 wrong  \u2205 missing  \u26a0 fell for distractor" }));
  }
}

function predValueFor(predRecords, recordIdx, slot) {
  if (!predRecords || !predRecords[recordIdx]) return undefined;
  return predRecords[recordIdx][slot];
}

function renderRecordsPanel(sample) {
  const panel = document.getElementById("records-panel");
  panel.innerHTML = "";
  const models = Object.keys(sample.preds || {}).sort();

  (sample.records || []).forEach((rec, ri) => {
    const slotNames = Object.keys(rec).filter((k) => k !== "main_entity");
    const table = el("table", { class: "record-table" });
    table.appendChild(el("caption", { text: rec.main_entity || `record ${ri + 1}` }));
    const headRow = el("tr", {}, [el("th", { text: "slot" }), el("th", { text: "gold" })]);
    for (const model of models) headRow.appendChild(el("th", { text: model }));
    table.appendChild(headRow);
    for (const slot of slotNames) {
      const gold = rec[slot];
      const row = el("tr", {}, [
        el("th", { text: slot }),
        gold ? el("td", { text: gold }) : el("td", { class: "null-val", text: "null" }),
      ]);
      for (const model of models) {
        const predRecords = sample.preds[model] && sample.preds[model].records;
        const sv = scorerVerdict(sample, model, ri, slot);
        if (sv) {
          const ok = sv.v === "ok" || sv.v === "null_ok";
          const title = { ok: "correct (scorer match, incl. partial span)", null_ok: "correctly null",
            wrong: "wrong value", missing: "missed (gold has a value)", spurious: "should be null" }[sv.v];
          row.appendChild(el("td", { class: ok ? "pred-ok" : "pred-bad", title, text: sv.pred == null ? "null" : String(sv.pred) }));
          continue;
        }
        const pv = predValueFor(predRecords, ri, slot);
        if (pv === undefined) row.appendChild(el("td", { class: "pred-missing", text: "\u2014" }));
        else if (pv === null || pv === "") row.appendChild(el("td", { class: gold ? "pred-bad" : "pred-ok", text: "null" }));
        else row.appendChild(el("td", { class: norm(pv) === norm(gold) ? "pred-ok" : "pred-bad", text: pv }));
      }
      table.appendChild(row);
    }
    panel.appendChild(table);
  });
}

function renderHeader(sample) {
  const header = document.getElementById("sample-header");
  header.innerHTML = "";
  header.appendChild(el("span", { class: "sid", text: sample.id }));
  header.appendChild(el("span", { class: "badge", text: sample.lang }));
  header.appendChild(el("span", { class: "badge", text: sample.format }));
  header.appendChild(el("span", { class: "badge", text: sample.schema_id }));
  if (sample.genre) header.appendChild(el("span", { class: "badge", text: sample.genre }));
  if (sample.tier) header.appendChild(el("span", { class: `badge tier-${sample.tier}`, text: sample.tier }));
  for (const hc of sample.hard_cases || []) {
    const desc = DATASET.hard_case_descriptions[hc] || DATASET.hard_case_descriptions[hc.replace(/c$/, "")] || "";
    header.appendChild(el("span", { class: "badge hc", title: desc, text: hc }));
  }
  for (const f of sample.flags || []) header.appendChild(el("span", { class: "badge flag", text: `\u2605 ${f}` }));
  for (const [model, p] of Object.entries(sample.preds || {}).sort()) {
    if (p.f1 == null) continue;
    const cls = p.f1 >= 0.99 ? "f1-ok" : p.f1 >= 0.7 ? "f1-mid" : "f1-bad";
    header.appendChild(el("span", { class: `badge ${cls}`, title: "sample slot F1", text: `${model}: ${p.f1.toFixed(2)}` }));
  }
}

function renderReasoningPanel(sample) {
  const panel = document.getElementById("reasoning-panel");
  panel.innerHTML = "";
  for (const [model, p] of Object.entries(sample.preds || {}).sort()) {
    const d = el("details");
    d.appendChild(el("summary", { text: `${model}: ${p.reasoning ? "reasoning + " : ""}raw output` }));
    if (p.reasoning) d.appendChild(el("pre", { class: "prompt-block", text: p.reasoning }));
    d.appendChild(el("pre", { class: "prompt-block", text: (p.raw || "").replace(/<think>[\s\S]*?(<\/think>|$)/, "").trim() }));
    panel.appendChild(d);
  }
}

// ---------- prompt chain view ----------

function renderPromptsView(sample) {
  const view = document.getElementById("prompts-view");
  view.innerHTML = "";
  const chain = DATASET.prompt_chains && DATASET.prompt_chains[sample.schema_id];
  if (!chain) {
    view.appendChild(el("p", { class: "muted", text: "No prompt chain available for this schema." }));
    return;
  }
  view.appendChild(el("h2", { text: "Background prompt" }));
  view.appendChild(el("pre", { class: "prompt-block", text: chain.background }));

  view.appendChild(el("h2", { text: "Extraction prompt chain" }));
  chain.stages.forEach((stage) => {
    const card = el("div", { class: "stage-card" });
    card.appendChild(el("h3", { text: stage.title }));
    const meta = el("div", { class: "filter-group" });
    for (const f of stage.added_fields) meta.appendChild(el("span", { class: "chip", text: f }));
    if (stage.uses_prev_stage) meta.appendChild(el("span", { class: "chip active", text: "uses previous stage's output" }));
    card.appendChild(meta);
    const details = el("details");
    details.appendChild(el("summary", { text: "Rendered prompt (one-shot example from this dataset)" }));
    details.appendChild(el("pre", { class: "prompt-block", text: stage.prompt_preview }));
    card.appendChild(details);
    view.appendChild(card);
  });
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("#view-tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.getElementById("doc-view").hidden = view !== "doc";
  document.getElementById("prompts-view").hidden = view !== "prompts";
}

document.querySelectorAll("#view-tabs .tab").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

// ---------- selection / bootstrap ----------

function selectSample(id) {
  const sample = DATASET.samples.find((s) => s.id === id);
  if (!sample) return;
  state.selectedId = id;
  document.getElementById("empty-state").hidden = true;
  document.getElementById("sample-view").hidden = false;
  renderList();
  renderHeader(sample);
  renderDoc(sample);
  renderLegend(sample);
  renderRecordsPanel(sample);
  renderReasoningPanel(sample);
  renderPromptsView(sample);
  setView(state.view);
  if (window.history && window.history.replaceState) window.history.replaceState(null, "", `#${encodeURIComponent(id)}`);
}

function refreshCurrent() {
  if (state.selectedId) selectSample(state.selectedId);
}

document.getElementById("toggle-decoys").addEventListener("change", refreshCurrent);

initDatasetSelector();
