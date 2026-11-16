# Label-noise audit: Qwen3.8-27B INT4 (and Qwen3-4B SFT) vs gold

**Question.** Base Qwen3.8-27B INT4 is imperfect on about 25% of docs: pilot 73 of 291 valid-JSON docs (slot-F1 0.966), grpo_dev 13 of 57 (F1 0.953). Is that label noise or real model errors?

**Answer.**
- **Pilot: mostly not model errors.** In a random sample of 25 of the 73 imperfect docs (seed 0), only 4 of 30 disagreements (13%) are real model errors. The rest split three ways:
  - 40%: the scorer rejects an equivalent span.
  - 27%: the slot is ambiguous.
  - 20%: gold is wrong or incomplete.
- **grpo_dev: real model errors.** All 13 imperfect docs were audited. 224 of 234 lost values are true model errors, and 203 of those come from 44 records in chunks whose thinking hit `max_tokens`, so the merge dropped them silently. Gold noise (a/b) is 0. The ambiguous cases are 9 slots in one doc.

Method: I read every disagreement against the doc text and the slot definition (no LLM judge). Diffs come from the scorer's own `align`/`match` (`build_diffs.py`). Verdicts are in `adjudicate.py` and F1 is in `compute_f1.py`. One line per disagreement is in `label_noise_audit.jsonl`.

Categories:
- a: gold wrong
- b: gold incomplete
- c: ambiguous
- d: model error (subtype)
- e: scorer matching issue

## 1. Counts by category: Qwen3.8-27B INT4

| set | audited | a gold_wrong | b gold_incomplete | c ambiguous | d model_error | e scorer_issue |
|---|---|---|---|---|---|---|
| pilot (25 of 73 imperfect docs) | 30 slots | 2 | 4 | 8 | 4 | 12 |
| grpo_dev (13 of 13 imperfect docs) | 31 slots + 44 missed records (234 values) | 0 | 0 | 9 (1 doc) | 21 slots + 44 records (224 values) | 1 |

Model-error subtypes:
- **grpo_dev:**
  - missed: 44 truncated records (203 values) plus 7 unresolved ditto marks
  - wrong value: 8 literal ditto copies ("همان"), 2 copy typos, 1 hallucinated
  - decoy: 3 (previous-status column)
- **pilot:** 2 missed, 2 wrong value (1 copy typo).

**Scorer issues (e)** are all span-boundary cases. The prediction keeps the gold head plus a qualifier, e.g. "suspended pending further review by …" vs "suspended", or "our depot in Boise" vs "Boise". `match()` rejects these because the substring ratio is below 0.6 and the token-subset ratio is below 0.33.

**Ambiguous (c):**
- **status synonym:** en-0-00042 (finalized vs signed)
- **event date vs slot definition:**
  - fa-0-00077: refit completion date
  - he-0-00090: inspection date
  - he-0-00038: "as of" date. Found by the date scan; all models agree with gold, so no error is scored.
- **unstated org roles:** en-0-00062 operator; fa-0-00046 supplier/client
- **status-as-location:** fa-0-00061
- **vague extra entity:** he-0-00041
- **ditto vs previous-value column:** grpo_dev fa-2-00233. "همان" in the client column sits next to a "previous client" column. All 4 models read it as "same as previous client", while gold resolves it from the row above (9 slots).

**Gold fixes (a/b):** 8 slots in 6 pilot docs; see `gold_patches.jsonl`.

| id | slot | gold → fix | why |
|---|---|---|---|
| en-0-00014 | location | Bristol industrial park → null | a: previous location (the car vanished) |
| he-0-00083 | location | חולון → מחסן | a: "הוחזר למחסן", so Holon is the previous location |
| en-0-00047 | date | 17 Jul 2025 → null | a: travel-calendar date; the workshop was cancelled (user finding, confirmed) |
| en-0-00024 | date | 26 Jan 2027 → null | a: structural-evaluation date, not the voyage date |
| fa-0-00052 r0–2 | destination | null → انبار مرکزی | b: the text says the listed shipments go to the central warehouse |
| he-0-00061 | status | null → מוצגת כעת לבדיקה עבור קונים פוטנציאליים | b: current status is stated |

## 2. Estimated noise rate (extrapolated)

| set | gold values | a+b (gold noise) | c (ambiguous) | e (scorer) | true 27B model error, share of disagreements |
|---|---|---|---|---|---|
| pilot | 2152 | 6 in sample → **~18 slots, about 0.9% of values; ~12 docs, about 4% of docs** (+ en-0-00047 type, invisible to disagreement audits) | 8 + 1 → **~24 slots (1.1%), ~20 docs (7%)** | 12 → ~35 slots | **13%** of INT4 disagreements (~12 values in the 73 docs) |
| grpo_dev | 2950 | **0** | 9 slots in 1 doc (0.3% of values, 1 of 60 docs) | 1 | **96%** of disagreements (224 of 234 values) |

Pilot figures are scaled by 73/25. Overall gold noise is low: about 1% wrong or missing values in pilot and about 0% in grpo_dev. The pilot shortfall is mostly convention issues (scorer spans, ambiguous definitions). It is a lower bound for errors that every model copies; see §5.

## 3. F1 recomputed with a/b/e fixed (valid-JSON docs; `f1_corrected.json`)

| model / set | base | a/b gold patched | a/b/e fixed (audited items only) | a/b/e fixed, extrapolated | also drop c |
|---|---|---|---|---|---|
| 27B INT4 pilot | 0.966 | 0.968 | **0.973** | **≈0.988** (×73/25) | 0.976 / ≈0.994 extrap |
| 27B INT4 grpo_dev | 0.953 | 0.953 | **0.953** | (exhaustive) | 0.956 |
| 4B SFT pilot | 0.958 | 0.958 | 0.959 | ≈0.960 (×80/40) | 0.960 |
| 4B SFT grpo_dev | 0.859 | 0.859 | **0.864** | (exhaustive) | 0.864 |
| Sonnet 5.5 pilot | 0.984 | 0.982 | 0.982 | - | - |
| Llama-3.3-70B pilot | 0.962 | 0.960 | 0.960 | - | - |

The pilot gold patches make Sonnet and Llama slightly *worse*: they copied the wrong dates and locations that gold had. On grpo_dev, the 27B shortfall is real. Without the 44 truncated records (an inference setting, not a model-quality issue), recall would rise from 0.918 to about 0.990.

## 4. Top model error types (for fine-tuning data)

1. **Missed values or records:**
   - 27B: chunk thinking hits `max_tokens` (12288) without `</think>`, so the chunk is dropped and its records disappear.
   - Unresolved ditto marks (`"`, 〃, "همان") and header/caption propagation ("accounts belong to בנק X", one client for the whole list).
   - 4B: whole records omitted (145 values on grpo_dev).
2. **Decoys and distractors:** previous-status, previous-location and insurer columns, planted distractors, and cross-record bleed. This is the dominant 4B error: 191 values on grpo_dev, 12 of 36 in the pilot sample.
3. **Wrong value:** the ditto word "همان" copied literally, copy typos and transliterations (מודיעין→"moidhin", نمایندگی→نماینده), and garbled/mixed-script output.

## 5. Qwen3-4B SFT section (scope A)

Scope: 4B errors on slots where **both** Sonnet 5.5 and Llama-70B match gold. That gives 421 cases on grpo_dev (all audited) and 80 on pilot (40 sampled, seed 0).
- **pilot, 40 manual:**
  - a: 2 (en-0-00047, en-0-00024). Here the 4B null is right and gold, Sonnet, Llama and INT4 are all wrong.
  - e: 2 (span)
  - d: 36, broken down as:
    - missed: 14 (record omitted 4, header propagation 2)
    - decoy: 12 (adjacent/insurer column, earlier status, cross-record, previous location, role swap)
    - wrong value: 9 (copy typo 5, hallucinated 2, role instead of name 1)
    - format: 1
- **grpo_dev, 421, rule-based with 12 spot-checked by reading (all consistent):**
  - d: 386, broken down as:
    - missed: 162 (record omitted 145; every chunk closed `</think>`, so these are not truncation)
    - decoy: 191 (planted distractor 131, cross-record 34, null slot filled 26)
    - wrong value: 33 (copy typo 12, ditto literal 7)
  - e: 35. The 4B writes "—"/"-" for empty table cells, and the scorer counts it as a filled value.
  - a/b: 0
- **User finding confirmed (en-0-00047).** Gold "17 Jul 2025" appears only as "after further review of the upcoming travel calendar for 17 Jul 2025 … the seminar … is officially cancelled". It is a travel date; the cancelled workshop has no stated date. The 4B (null) was correct, while gold and the models agreeing with it were wrong.
- **Same-pattern scan.** I read all 26 pilot gold dates whose ±150-char window contains cancel, postpone, travel, "was supposed", originally or proposed (he/fa/en keywords). Results:
  - Gold wrong: en-0-00047, plus en-0-00024 (an inspection date).
  - Ambiguous: he-0-00038 ("as of" status date; all models copy it), plus he-0-00090 and fa-0-00077 (found earlier).
  - Consistent: the other 21. Cancelled events keep their scheduled date, e.g. he-0-00098 "היה אמור להתקיים ביום 1 באפריל 2025 … בוטל", where the gold is correct and the 4B miss is real. Postponed events use the new date (fa-0-00012, en-0-00050), and "planned forecast vs official date" picks the official one (fa-0-00055).
  - Because every model copies this kind of error, only a model that abstains (the 4B here) exposes it.

## 6. Scorer issues (e): described, not changed

1. **Span leniency:** `re_synth/score.py` `match()` lines 106–135. The substring rule needs `len(short)/len(long) >= 0.6` (line 131) and the token-subset rule needs `>= 0.33` (line 135), so status and location values carrying a qualifier fail. Options:
   - Accept the prediction when the gold value is a whole-token prefix or suffix of it, for status and location types.
   - Or allow list-of-alternates in gold. Currently a list gold requires *all* its values to be matched (lines 109–113), so a list cannot express alternatives.
2. **Placeholder nulls:** `re_synth/score.py` `_records()` line 214. Only `"", "null", "None", [], "N/A"` map to None, so "-", "—" and "–" count as predicted values and score as FP (35 4B cases on grpo_dev). Fix: add the dash tokens to the null list.
3. **Hidden truncation (inference/merge, not matching):** `re_synth/chunking.py` `merge_raws()` lines 306–318 skips an unparseable chunk. A doc whose long chunk hit `max_tokens` therefore still counts as `valid_json` while its records vanish (27B grpo_dev: he-2-00294, fa-2-00147, fa-2-00207, fa-2-00233). Raise `max_tokens` or use smaller chunks, and report truncated chunks.

## 7. Recommendation

- **Correct** the 8 a/b slots. Corrected copies are `data/pilot/export/pilot_all_corrected.jsonl` and `data/pilot/samples_corrected.jsonl`; the originals are untouched and grpo_dev needs no patch.
- **Remove or neutralize** the c items listed in `remove_ids.json`: drop grpo_dev fa-2-00233, or regenerate it without a ditto next to a previous-value column, and null or drop the 8 ambiguous pilot slots. Also fix the generator so date slots never come from travel or inspection sentences.
- **Fix the scorer** (dash nulls, head/suffix span leniency). On pilot, scorer issues hide more than gold noise does: about 35 slots vs about 18.
- **Fine-tuning data** should target ditto/header propagation, previous-value/insurer decoy columns, and verbatim copying (no transliteration). For 27B inference, raise `max_tokens` or shrink chunks.

Files (`outputs/audit/`):
- `build_diffs.py`, `diffs_*.jsonl`: diff builder and per-slot diffs
- `adjudicate.py` → `label_noise_audit.jsonl`, `gold_patches.jsonl`, `remove_ids.json`, plus the corrected copies
- `compute_f1.py` → `f1_corrected.json`

INT4 predictions were read from `/tmp/int4` (set `INT4_DIR` to override); they are not committed.
