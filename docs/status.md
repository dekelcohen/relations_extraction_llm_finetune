# Project status

Living summary of where this project stands and what's next. For full detail see the
[research survey](research/entity_relation_extraction.md) and the
[plan](plans/synth_dataset_and_finetune.md) (execution log in plan §10).

## Where we are
Research + plan are done. The synthetic pipeline, scorer, mining, SFT/GRPO training
scripts and eval harness are implemented and smoke-tested. A **300-sample pilot**
(he/fa/en, 8 schemas, text/HTML/email) has been generated and scored against
Llama-3.3-70B (weak) vs Claude Sonnet 5.5 (strong) baselines — see
[plan §10](plans/synth_dataset_and_finetune.md#10-execution-status-pilot-run-2026-11-10)
for the full table. Headline: weak/strong slot-F1 gap is real but small (0.958 vs
0.984); most pilot samples are "easy" (only 42/298 mined as hard).

## Immediate next step: manual review of the pilot testset
**Before scaling the build (3000/lang) or hardening further, the 300-sample pilot
needs a manual pass to check it's actually representative of the target domain** —
i.e. does synthetic text/formatting resemble the private company documents this
model will run on in production, not just generic Wikipedia/news prose?

Review checklist (per sample, sampled across languages/formats/schemas):
- [ ] **Format realism** — do the HTML tables, email headers and plain-text layouts
      look like real internal docs (invoices, shipment notices, HR records, asset
      logs), or do they read as generic/encyclopedic?
- [ ] **Entity/relation realism** — are main-entity types, slot values (IDs, dates,
      locations, statuses, orgs) and their typical co-occurrence patterns close to
      what shows up in the user's real documents?
- [ ] **Distractor realism** — are the planted decoys (irrelevant nearby dates/
      locations/orgs, H1–H12 hard cases) the kind of ambiguity that actually occurs
      in real docs, or are they arbitrary/synthetic-looking?
- [ ] **Language fidelity (he/fa)** — natural phrasing, correct date formats
      (Hebrew/Jalali), no translation artifacts or broken diacritics/gershayim.
- [ ] **Noise/boilerplate** — real docs often have disclaimers, signatures, repeated
      headers, OCR noise; check whether the synthetic set under- or over-represents
      this compared to the private corpus.

Output of the review: a short gap list (what's missing or unrealistic), used to
drive the next round of schema/hardcase tuning — not a pass/fail gate.

## Then: how to make the testset harder
The pilot's own "Next" section (plan §10) already lists the first round of
hardening, to be combined with findings from the manual review above:
- More records per document (3–6 instead of ~1–2), so candidate-pair disambiguation
  matters more.
- More decoys per slot (2–3 instead of 1).
- Longer emails/threads (multi-message, nested quoting) and longer HTML tables.
- More H6 (subtitle → list, N lines below) and H11 (near-but-unrelated-date) cases,
  where Llama was weakest (slot F1 0.895 / 0.970 vs Sonnet ≈ 0.99).
- If the manual review surfaces domain-specific patterns (e.g. recurring document
  templates, house-style abbreviations, specific ID formats) not currently covered
  by `re_synth/schemas.py` / `re_synth/hardcases.py`, add them as new schemas/hard
  cases rather than only scaling existing ones.
- Re-run mining after each hardening round; target ≥ 30% hard-tier yield (vs ~14%
  in the pilot) before scaling to the full 3000/lang build.

## Longer-term next steps (unchanged from plan §10)
1. Scale the build to 3000 samples/language once hardening above is applied.
2. Convert Re-DocRED and add the P1 public datasets.
3. Run SFT → GRPO on 4×A100 (README §3).
4. Evaluate on `grpo_test` + the forgetting suite (`eval/forgetting.sh`).

## Open questions (carried from plan §9, still unanswered)
1. Exact main-entity types and slots of interest beyond vehicle/person/org/shipment
   — real example documents (even 10, redacted if needed) would let the manual
   review above be done against actual private-doc patterns instead of guesswork.
2. Output detail: verbatim strings only, or normalized dates/IDs too? Evidence
   spans required?
