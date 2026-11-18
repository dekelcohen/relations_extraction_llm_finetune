#!/bin/bash
# Rebuild outputs/report/pilot_compare.html: pilot comparison + grpo_dev and test_long sections.
# Qwen3.8-27B columns (fp8 / bnb4 base, ft-fp8 / ft-bnb4) are added when their preds exist.   Run from the repo root (any env with the repo deps).
set -e; cd "$(dirname "$0")/.."
E=outputs/eval; P=data/pilot; D=data/stage_2_RL_GRPO; T=data/test_long
VURL=${VIEWER_URL:-http://10.45.93.247:8111/}   # external host so links work off-box too; override via VIEWER_URL=...
add() { [ -f "$2" ] && echo "--model $1=$2"; }   # label path
B27="qwen3.8-27b-fp8 qwen3.8-27b-int4 qwen3.8-27b-int4-medium qwen3.8-27b-bnb4 qwen3.8-27b-ft-int4 qwen3.8-27b-ft-int4-medium qwen3.8-27b-ft-stage2-int4 qwen3.8-27b-ft-stage2-int4-medium qwen3.8-27b-ft-fp8 qwen3.8-27b-ft-bnb4"
n27() { for l in $B27; do case $1 in pilot) f=$E/$l-think.preds.jsonl ;; *) f=$E/$l.$1.preds.jsonl ;; esac; add $l $f; done; }
TH="vLLM, thinking, T=0.6 top_p 0.95 top_k 20, max 5120 tok, 1 sample"
NOTES=(--note "sonnet-5.5=anthropic/claude-sonnet-5.5 via OpenRouter, T=0" --note "llama-3.3-70b=meta-llama/llama-3.3-70b-instruct via OpenRouter, T=0"
  --note "qwen3-4b=Qwen/Qwen3-4B-AWQ, $TH" --note "qwen3-4b-ft=same + QLoRA adapter (r32, 2 ep on 507 he/fa teacher-CoT rows; trained on bnb-NF4 base)"
  --note "qwen3-4b-nothink=Qwen/Qwen3-4B-AWQ, enable_thinking=False, T=0.7 top_p 0.8 top_k 20, max 2048 tok, 1 sample"
  --note "qwen3-4b-ft-nothink=same as qwen3-4b-nothink + the QLoRA adapter (trained WITH thinking: out-of-distribution mode)"
  --note "qwen3-4b-grpo=same as qwen3-4b-ft + 20 GRPO steps (strict reward, 40 prompts x G=4, final adapter), same thinking settings"
  --note "qwen3.8-27b-fp8=BASE Qwen/Qwen3.8-27B-FP8 (8-bit weights, vLLM Marlin), $TH"
  --note "qwen3.8-27b-int4=BASE RedHatAI/Qwen3.8-27B-INT4 (W4A16), bf16 KV cache, Triton attention, max_num_seqs 48, thinking T=0.6 top_p 0.95 top_k 20, **max 12288 tok** (max_model_len 16384; fp8 column used 5120), 1 sample"
  --note "qwen3.8-27b-ft-int4=27B + the same QLoRA adapter served on the INT4 base (vLLM LoRA), same settings as qwen3.8-27b-int4"
  --note "qwen3.8-27b-int4-medium=same as qwen3.8-27b-int4 but chat_template reasoning_effort=medium (default is xhigh), all prompts"
  --note "qwen3.8-27b-ft-int4-medium=same as qwen3.8-27b-ft-int4 but reasoning_effort=medium (adapter was trained at the default xhigh instruction)"
  --note "qwen3.8-27b-ft-stage2-int4=27B-ft-stage-2: stage-1 adapter continued (lr 5e-5, 2 ep, r32) on rejection-sampled self-distilled base-INT4-medium traces (data/sft_stage2/train_selfdistill.jsonl), served on INT4, default (xhigh) instruction"
  --note "qwen3.8-27b-ft-stage2-int4-medium=same adapter at reasoning_effort=medium (the instruction it was trained under)"
  --note "qwen3.8-27b-bnb4=BASE unsloth/Qwen3.8-27B-unsloth-bnb-4bit (vLLM bitsandbytes), $TH"
  --note "qwen3.8-27b-ft-fp8=27B + QLoRA adapter (r32, 2 ep, same data/recipe as qwen3-4b-ft; trained on the bnb-4bit base) served on the FP8 base, $TH"
  --note "qwen3.8-27b-ft-bnb4=27B + the same QLoRA adapter served on its training base (bnb-4bit), $TH"
  --note "qwen3-4b-ft-stage2=qwen3-4b-ft + stage-2 SFT continued on data/sft_stage2/train_selfdistill.jsonl (same recipe/data as qwen3.8-27b-ft-stage2-int4, lr 5e-5, 2 ep), same thinking settings")
python eval/compare_report.py --samples $P/samples.jsonl --model sonnet-5.5=$P/preds_strong.jsonl --model llama-3.3-70b=$P/preds_weak.jsonl \
  --model qwen3-4b=$E/qwen3-4b-think.preds.jsonl --model qwen3-4b-ft=$E/qwen3-4b-ft-think.preds.jsonl \
  --model qwen3-4b-nothink=$E/qwen3-4b-nothink.preds.jsonl --model qwen3-4b-ft-nothink=$E/qwen3-4b-ft-nothink.preds.jsonl \
  --model qwen3-4b-grpo=$E/qwen3-4b-grpo-think.preds.jsonl --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq-think.preds.jsonl $(n27 pilot) \
  --base qwen3-4b --ft qwen3-4b-ft --frontier sonnet-5.5 --oss70 llama-3.3-70b "${NOTES[@]}" --viewer_dir $P --viewer_url $VURL \
  --err27 qwen3.8-27b-ft-int4 --err27_ref qwen3.8-27b-int4 --err27_ref qwen3.8-27b-fp8 \
  --section_models qwen3.8-27b-fp8,qwen3.8-27b-int4,qwen3.8-27b-int4-medium,qwen3.8-27b-ft-int4,qwen3.8-27b-ft-stage2-int4-medium,qwen3-4b-ft,qwen3-4b-grpo,qwen3-4b-ft-stage2,llama-3.3-70b,sonnet-5.5 --out outputs/report/pilot_compare.html
python eval/append_section.py --report outputs/report/pilot_compare.html --id grpo_dev \
  --title "Stage-2 grpo_dev (60 held-out hard docs, doc-level, chunked inference)" --rows $D/grpo_dev.jsonl \
  --model qwen3-4b-ft=$E/sft.grpo_dev.preds.jsonl --model grpo-ckpt10=$E/grpo-mid.grpo_dev.preds.jsonl \
  --model qwen3-4b-grpo=$E/grpo-final.grpo_dev.preds.jsonl --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.grpo_dev.preds.jsonl \
  --model sonnet-5.5=$D/preds_strong.jsonl --model llama-3.3-70b=$D/preds_weak.jsonl \
  $(n27 grpo_dev) --note "lenient scorer; used for checkpoint selection (final adapter chosen by dev slot-F1)"
python eval/append_section.py --report outputs/report/pilot_compare.html --id test_long \
  --title "Held-out test_long (80 realistic long hard docs, he+fa, chunked inference)" --rows $T/test_long.jsonl \
  --model qwen3-4b-ft=$E/sft.test_long.preds.jsonl --model qwen3-4b-grpo=$E/grpo.test_long.preds.jsonl \
  --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.test_long.preds.jsonl \
  --model llama-3.3-70b=$T/preds_weak.jsonl --model sonnet-5.5=$T/preds_strong.jsonl $(n27 test_long) \
  --note "never used for training or checkpoint selection; lenient scorer"
python eval/append_section.py --report outputs/report/pilot_compare.html --id hard_inss \
  --title "hard_inss dev (39 hard INSS-hosted docs, he+fa, up to 9.6k chars, chunked inference)" --rows data/hard_inss/dev.jsonl \
  --model qwen3-4b-grpo=$E/grpo.hard_inss.preds.jsonl --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.hard_inss.preds.jsonl \
  --model llama-3.3-70b=data/hard_inss/preds_weak.jsonl $(n27 hard_inss) \
  --note "never used for training; lenient scorer; 4B-ft-s1/Sonnet not run on this set"
# 27B-ft-stage-1 (= qwen3.8-27b-ft-int4) error analysis for grpo_dev and test_long (pilot's is inside the main Error analysis)
ea() { # id title rows ft_label grpo_label strong_weak_dir viewer_dataset
  python eval/append_errors.py --report outputs/report/pilot_compare.html --id err27_$1 --title "Error analysis: 27B-ft-stage-1 (qwen3.8-27b-ft-int4) on $2" --rows $3 \
    --model qwen3.8-27b-fp8=$E/qwen3.8-27b-fp8.$1.preds.jsonl --model qwen3.8-27b-int4=$E/qwen3.8-27b-int4.$1.preds.jsonl \
    --model qwen3.8-27b-ft-int4=$E/qwen3.8-27b-ft-int4.$1.preds.jsonl --model qwen3-4b-ft=$E/${4}.$1.preds.jsonl \
    --model qwen3-4b-grpo=$E/${5}.$1.preds.jsonl --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.$1.preds.jsonl \
    --model llama-3.3-70b=${6}/preds_weak.jsonl --model sonnet-5.5=${6}/preds_strong.jsonl \
    --em qwen3.8-27b-ft-int4 --ref qwen3.8-27b-int4 --ref qwen3.8-27b-fp8 --viewer_url $VURL --viewer_dataset $7
}
ea grpo_dev "grpo_dev" $D/grpo_dev.jsonl sft grpo-final $D stage_2_RL_GRPO
ea test_long "test_long" $T/test_long.jsonl sft grpo $T test_long
# 4B-ft-stage-2 (self-distill) own error analysis, focused on it (em=qwen3-4b-ft-stage2) -- was previously only a
# reference column inside the 27B-focused err27_* sections above.
python eval/append_errors.py --report outputs/report/pilot_compare.html --id err4b_test_long \
  --title "Error analysis: 4B-ft-stage-2 (self-distill) on test_long" --rows $T/test_long.jsonl \
  --model qwen3-4b-ft=$E/sft.test_long.preds.jsonl --model qwen3-4b-grpo=$E/grpo.test_long.preds.jsonl \
  --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.test_long.preds.jsonl \
  --model qwen3.8-27b-ft-stage2-int4-medium=$E/qwen3.8-27b-ft-stage2-int4-medium.test_long.preds.jsonl \
  --model llama-3.3-70b=$T/preds_weak.jsonl --model sonnet-5.5=$T/preds_strong.jsonl \
  --em qwen3-4b-ft-stage2 --ref qwen3-4b-grpo --ref qwen3-4b-ft --viewer_url $VURL --viewer_dataset test_long
# hard_inss only has the medium-effort 27B runs + 4B-grpo/ft-stage2 + Llama (no fp8, no non-medium 27B, no Sonnet, no 4B-ft-s1)
python eval/append_errors.py --report outputs/report/pilot_compare.html --id err27_hard_inss \
  --title "Error analysis: 27B-ft-stage-1 (qwen3.8-27b-ft-int4-medium) on hard_inss" --rows data/hard_inss/dev.jsonl \
  --model qwen3.8-27b-int4-medium=$E/qwen3.8-27b-int4-medium.hard_inss.preds.jsonl \
  --model qwen3.8-27b-ft-int4-medium=$E/qwen3.8-27b-ft-int4-medium.hard_inss.preds.jsonl \
  --model qwen3.8-27b-ft-stage2-int4-medium=$E/qwen3.8-27b-ft-stage2-int4-medium.hard_inss.preds.jsonl \
  --model qwen3-4b-grpo=$E/grpo.hard_inss.preds.jsonl --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.hard_inss.preds.jsonl \
  --model llama-3.3-70b=data/hard_inss/preds_weak.jsonl \
  --em qwen3.8-27b-ft-int4-medium --ref qwen3.8-27b-int4-medium --viewer_url $VURL --viewer_dataset hard_inss
python eval/append_errors.py --report outputs/report/pilot_compare.html --id err4b_hard_inss \
  --title "Error analysis: 4B-ft-stage-2 (self-distill) on hard_inss" --rows data/hard_inss/dev.jsonl \
  --model qwen3-4b-grpo=$E/grpo.hard_inss.preds.jsonl --model qwen3-4b-ft-stage2=$E/qwen3-4b-ft-stage2-awq.hard_inss.preds.jsonl \
  --model qwen3.8-27b-ft-stage2-int4-medium=$E/qwen3.8-27b-ft-stage2-int4-medium.hard_inss.preds.jsonl \
  --model llama-3.3-70b=data/hard_inss/preds_weak.jsonl \
  --em qwen3-4b-ft-stage2 --ref qwen3-4b-grpo --viewer_url $VURL --viewer_dataset hard_inss
python eval/strong_gap.py --report outputs/report/pilot_compare.html --viewer_url $VURL   # 27B errors where Sonnet AND Llama are right
python eval/build_summary.py --report outputs/report/pilot_compare.html   # top summary table (all sets x models) + nav dropdown
