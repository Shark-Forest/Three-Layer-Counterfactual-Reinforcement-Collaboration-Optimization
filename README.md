# Experiment 28: Proposal-Review Controller on GSM8K

This directory is a self-contained copy of experiment 28 from `lcs/MAS/three_level`.
The original `three_level` directory was left unchanged.

The goal of this copy is to make the code easier to read, reproduce, and upload for review.
It keeps the experiment-28 code path, the GSM8K cache, and the checkpoints/results that had
already been produced.

## What Changed In This Copy

Only the dataset selection was changed for future runs:

- training now uses the full official GSM8K `train` split
- testing now uses the full official GSM8K `test` split
- validation is empty by default
- the old 100-sample cap is removed

The copied historical results under `runs/` are still the already-produced 100-sample experiment
28 results. New runs launched from this directory will use the full GSM8K train/test splits unless
you explicitly set sample caps with environment variables.

## Directory Layout

```text
end/
  README.md
  requirements.txt
  run_priority_diagnostics_28.py
  run_all_28.py
  eval_current_strategy_sample100_28.py
  eval_non_train_gsm8k_28.py
  src28/
    config.py
    data_loader.py
    gspo_verl.py
    model_loader.py
    parallel_runtime.py
    cfr_core.py
    metrics_logger.py
    verifier.py
    plotter.py
  data/
    GSM8K / Hugging Face / ModelScope cache copied from three_level
  runs/
    priority_diagnostics_20260428T143412Z/
    priority_diagnostics_20260428T144823Z/
```

The large base model cache from `three_level/models` was not copied. It is about 15 GB and should
be re-downloaded or supplied via the normal ModelScope/Hugging Face cache mechanism. A small
`models/datasets` cache may be created when checking/loading GSM8K; it is not the base model.
The generated experiment checkpoints are included under `runs/.../checkpoints`.

## Main Experiment

The main spec is:

```text
28_single_pending_neg1_replace_round5
```

It is defined in `run_priority_diagnostics_28.py`.

Important settings:

- architecture: controller + proposer + reviewer
- middle action schema: `proposal_review`
- train rounds: 5
- inference rounds: 5
- controller regret update mode: `selected_only`
- realized branch selection: `first`
- proposer/reviewer inner policies: GSPO with LoRA
- evaluation protocols:
  - `controller_stochastic`: average-strategy sampling controller
  - `controller_greedy`: average-strategy argmax controller

## Dataset Behavior

`src28/data_loader.py` now loads GSM8K as:

```text
train = official GSM8K train split
val   = []
test  = official GSM8K test split
```

Expected GSM8K sizes are normally:

```text
train: 7473
test:  1319
```

`run_priority_diagnostics_28.py` no longer defaults to 100 training/test samples. These environment
variables are still supported for smoke tests:

```text
PRIORITY_DIAG_TRAIN_SAMPLES=100
PRIORITY_DIAG_VAL_SAMPLES=100
PRIORITY_DIAG_TEST_SAMPLES=100
```

Unset, `0`, `all`, `full`, or `none` means no cap.

## Proposal-Review Protocol

Odd rounds are proposal rounds and even rounds are review rounds.

When there is no pending candidate, the proposer must create the initial pending solution.
When there is a pending candidate, the controller chooses between:

```text
keep_pending
refresh_pending
```

The reviewer only judges the current pending solution. It does not solve the problem again.

In experiment 28, a refresh can replace the current pending candidate only when the pending net
vote score is `<= -1`. This is the key difference from the immediately preceding variants.

## Controller State

The middle controller sees a small discrete state. It does not see the question text, ground truth,
or raw proposer/reviewer text.

Without a pending candidate:

```python
("controller", "mode=bootstrap")
```

With a pending candidate:

```python
(
    "controller",
    "pending_score_bucket=score0 or score1",
    "proposal_time_bucket=last or not_last",
)
```

`pending_score_bucket=score1` means pending vote score is at least 1. `score0` includes zero and
negative scores. `proposal_time_bucket=last` means there is only one proposal opportunity left.

## Prompts

The reviewer prompt is `PI0_PROMPT_PROPOSAL_REVIEW`:

```text
{context}

You are a careful reasoning assistant.
Check whether the current pending final answer is correct for the original math problem.
First line: RIGHT or WRONG.
Second line: one short reason.
Do not write a new solution or a new final answer.

Review:
```

The initial proposer prompt is `PI1_PROMPT_PROPOSAL_REVIEW`:

```text
{context}

Solve the original math problem directly.
If there is a pending solution or short review feedback, use it only as a hint.
Keep the reasoning concise.
End with exactly one final line: Final answer: <number>.

Solution:
```

The refresh proposer prompt is `PI1_PROMPT_PROPOSAL_REVIEW_REFRESH`:

```text
{context}

Solve the original math problem directly.
Use the current pending solution and the latest review only as hints.
If the latest review says WRONG, fix the decisive mistake and recompute.
If the latest review says RIGHT, keep the answer unless you find a clear error.
Keep the reasoning concise.
End with exactly one final line: Final answer: <number>.

Solution:
```

The context contains the original question, the current pending solution if one exists, and the
latest valid review feedback during proposal rounds.

## Historical Copied Results

The completed copied run is:

```text
runs/priority_diagnostics_20260428T144823Z/28_single_pending_neg1_replace_round5
```

It contains:

- `metadata.json`
- `logs/live_accuracy.jsonl`
- `logs/policy_stack_candidate_dump.jsonl`
- `checkpoints/latest_checkpoint.txt`
- checkpoint directories `sample_000010` through `sample_000100`
- LoRA policy checkpoints for `agent0_pi0.pt` and `agent0_pi1.pt`

The final 100-sample results copied from the original run were:

```text
training final round accuracies:      [0.83, 0.83, 0.84, 0.84, 0.85]
controller_stochastic test accuracy:  [0.83, 0.83, 0.83, 0.83, 0.83]
controller_greedy test accuracy:      [0.83, 0.83, 0.83, 0.83, 0.83]
```

Again, these are historical 100-sample results. A fresh run from this directory uses full GSM8K
train/test by default.

## Running A Full Experiment

From this directory:

```bash
cd /mnt/paper2any/lcs/MAS/end
conda activate lcs-metax
python run_priority_diagnostics_28.py
```

The script defaults to `28_single_pending_neg1_replace_round5`, so no `--experiments` argument is
needed. You can pass it explicitly if desired:

```bash
python run_priority_diagnostics_28.py --experiments 28_single_pending_neg1_replace_round5
```

For a quick smoke run on 100 train/test samples:

```bash
PRIORITY_DIAG_TRAIN_SAMPLES=100 \
PRIORITY_DIAG_TEST_SAMPLES=100 \
python run_priority_diagnostics_28.py
```

Checkpoint frequency is controlled by:

```text
PRIORITY_DIAG_CHECKPOINT_EVERY_SAMPLES
```

The inherited default is 10. For full GSM8K training, consider increasing it before launching a
new long run to avoid producing hundreds of large checkpoint directories.

## Evaluating An Existing Checkpoint

Evaluate the copied `sample_000100` checkpoint on the full test split:

```bash
python eval_current_strategy_sample100_28.py \
  --exp-dir runs/priority_diagnostics_20260428T144823Z/28_single_pending_neg1_replace_round5 \
  --checkpoint-name sample_000100 \
  --split test
```

Evaluate only the first 100 examples:

```bash
python eval_current_strategy_sample100_28.py \
  --exp-dir runs/priority_diagnostics_20260428T144823Z/28_single_pending_neg1_replace_round5 \
  --checkpoint-name sample_000100 \
  --split test \
  --test-samples 100
```

The script name still contains `sample100` because it was copied from the original experiment, but
in this directory its default is full-split evaluation.

## GitHub Upload Notes

The checkpoint files are large. Several `*.pt` files are about 145 MB each, which is above GitHub's
normal per-file limit. Use Git LFS for checkpoints:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.safetensors"
git add .gitattributes
```

If the reviewer only needs code and logs, keep checkpoints out of the repository and share them via
external storage. If the reviewer needs exact reproduction from `sample_000100`, keep the copied
checkpoint tree and use Git LFS or another artifact store.

## Reproducibility Notes

- The copied checkpoint metadata still records `total_samples=100`, because it comes from the
  historical 100-sample run.
- A new full-data run will create checkpoint metadata with the full train size.
- The base model is not vendored in this folder. The default model is
  `LLM-Research/Phi-3-mini-4k-instruct`, configurable through `MAS_MODEL_SCOPE`.
- CUDA is expected for normal runs. CPU execution is supported only as a fallback and is impractical
  for this setup.
