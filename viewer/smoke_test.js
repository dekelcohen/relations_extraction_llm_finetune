// Headless smoke test for the viewer: loads index.html via jsdom, checks that each
// format renders without throwing, that highlight marks actually got inserted, that the
// static-data.js fallback kicks in when no dynamic server is reachable, that the per-model
// prediction overlay renders verdict badges, and that the Prompt-chain tab renders stage cards.
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

async function main() {
  const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf-8");
  const dataJs = fs.readFileSync(path.join(__dirname, "data.js"), "utf-8");
  const appJs = fs.readFileSync(path.join(__dirname, "app.js"), "utf-8");

  const dom = new JSDOM(html, { runScripts: "outside-only", url: "http://localhost/" });
  const { window } = dom;
  // No `fetch` in this jsdom window -> initDatasetSelector()'s dynamic API call throws and it
  // must fall back to the static `DATASET` global baked into data.js by export_viewer_data.py.
  delete window.fetch;
  window.eval(dataJs);
  window.eval(appJs);
  await window.initDatasetSelector(); // app.js already kicked this off once; await it again to be sure it settled

  const dataset = window.DATASET;
  console.log(`DATASET.samples: ${dataset.samples.length}`);
  if (!dataset.samples.length) throw new Error("static fallback did not populate DATASET.samples");

  const selectOptions = window.document.querySelectorAll("#dataset-select option");
  if (selectOptions.length !== 1 || !selectOptions[0].textContent.includes("static"))
    throw new Error("dataset-select did not fall back to a single 'static data.js' option");

  const byFormat = {};
  for (const s of dataset.samples) byFormat[s.format] = byFormat[s.format] || s.id;

  for (const [fmt, id] of Object.entries(byFormat)) {
    window.selectSample(id);
    const container = window.document.getElementById("doc-container");
    const marks = container.querySelectorAll("mark");
    const mainMarks = container.querySelectorAll("mark.ent-main");
    if (marks.length === 0) throw new Error(`format=${fmt} sample=${id}: no <mark> elements rendered`);
    if (mainMarks.length === 0) throw new Error(`format=${fmt} sample=${id}: no main-entity mark rendered`);
    console.log(`OK format=${fmt} id=${id}: ${marks.length} marks (${mainMarks.length} main-entity)`);
  }

  // toggle decoys off/on, make sure nothing throws
  const anyWithDecoys = dataset.samples.find((s) => s.decoys && s.decoys.length);
  if (anyWithDecoys) {
    window.selectSample(anyWithDecoys.id);
    window.document.getElementById("toggle-decoys").checked = false;
    window.document.getElementById("toggle-decoys").dispatchEvent(new window.Event("change"));
    const noDecoyMarks = window.document.querySelectorAll("mark.ent-decoy").length;
    if (noDecoyMarks !== 0) throw new Error("toggle-decoys=false still rendered decoy marks");
    window.document.getElementById("toggle-decoys").checked = true;
    window.document.getElementById("toggle-decoys").dispatchEvent(new window.Event("change"));
    console.log(`OK decoy toggle for ${anyWithDecoys.id}`);
  }

  // Per-model prediction overlay: records table should always show a column per model, and
  // toggling a model "on" should add verdict badges next to gold/decoy marks; toggling it back
  // off should remove them again.
  const models = Object.keys(window.MODEL_CODES || {});
  console.log(`models discovered: ${models.join(", ") || "(none)"}`);
  if (models.length) {
    const sampleWithPreds = dataset.samples.find((s) => s.preds && Object.keys(s.preds).length);
    if (!sampleWithPreds) throw new Error("models discovered but no sample has a preds payload");
    window.selectSample(sampleWithPreds.id);

    const headerCols = window.document.querySelectorAll("table.record-table th").length;
    if (headerCols === 0) throw new Error("records panel did not render any columns");

    const model = models[0];
    window.state.activeModels.add(model);
    window.refreshCurrent();
    const badgesOn = window.document.querySelectorAll("sup.verdict").length;
    if (badgesOn === 0) throw new Error(`toggling model "${model}" on produced no verdict badges`);

    window.state.activeModels.delete(model);
    window.refreshCurrent();
    const badgesOff = window.document.querySelectorAll("sup.verdict").length;
    if (badgesOff !== 0) throw new Error(`toggling model "${model}" off left ${badgesOff} verdict badges behind`);

    console.log(`OK per-model overlay for ${sampleWithPreds.id}: ${badgesOn} badge(s) when "${model}" active`);
  }

  // Prompt-chain tab: switching view should render at least one stage card for a schema that has one.
  const sampleWithChain = dataset.samples.find((s) => dataset.prompt_chains && dataset.prompt_chains[s.schema_id]);
  if (sampleWithChain) {
    window.selectSample(sampleWithChain.id);
    window.setView("prompts");
    const stageCards = window.document.querySelectorAll("#prompts-view .stage-card").length;
    const promptBlocks = window.document.querySelectorAll("#prompts-view pre.prompt-block").length;
    if (stageCards === 0) throw new Error("prompt-chain view rendered no stage cards");
    if (promptBlocks === 0) throw new Error("prompt-chain view rendered no prompt blocks");
    if (window.document.getElementById("prompts-view").hidden) throw new Error("prompts-view should be visible after setView('prompts')");
    if (!window.document.getElementById("doc-view").hidden) throw new Error("doc-view should be hidden after setView('prompts')");
    window.setView("doc");
    console.log(`OK prompt-chain view for ${sampleWithChain.id}: ${stageCards} stage card(s)`);
  } else {
    console.warn("no sample with a prompt chain found; skipping prompt-chain view check");
  }

  console.log("ALL SMOKE TESTS PASSED");
}

main().catch((e) => {
  console.error("SMOKE TEST FAILED:", e);
  process.exit(1);
});
