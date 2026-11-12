// Headless smoke test for the viewer: loads index.html via jsdom, checks that each
// format renders without throwing and that highlight marks actually got inserted.
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

async function main() {
  const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf-8");
  const dataJs = fs.readFileSync(path.join(__dirname, "data.js"), "utf-8");
  const appJs = fs.readFileSync(path.join(__dirname, "app.js"), "utf-8");

  const dom = new JSDOM(html, { runScripts: "outside-only", url: "http://localhost/" });
  const { window } = dom;
  window.eval(dataJs);
  window.eval(appJs);

  const dataset = window.DATASET;
  console.log(`DATASET.samples: ${dataset.samples.length}`);

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

  // toggle decoys off/on and predictions on, make sure nothing throws
  const anyWithDecoys = dataset.samples.find((s) => s.decoys && s.decoys.length);
  if (anyWithDecoys) {
    window.selectSample(anyWithDecoys.id);
    window.document.getElementById("toggle-decoys").checked = false;
    window.document.getElementById("toggle-decoys").dispatchEvent(new window.Event("change"));
    const noDecoyMarks = window.document.querySelectorAll("mark.ent-decoy").length;
    if (noDecoyMarks !== 0) throw new Error("toggle-decoys=false still rendered decoy marks");
    window.document.getElementById("toggle-decoys").checked = true;
    window.document.getElementById("toggle-preds").checked = true;
    window.document.getElementById("toggle-preds").dispatchEvent(new window.Event("change"));
    const predCols = window.document.querySelectorAll("table.record-table th").length;
    if (predCols === 0) throw new Error("predictions panel did not render columns");
    console.log(`OK decoy toggle + predictions panel for ${anyWithDecoys.id}`);
  }

  console.log("ALL SMOKE TESTS PASSED");
}

main().catch((e) => {
  console.error("SMOKE TEST FAILED:", e);
  process.exit(1);
});
