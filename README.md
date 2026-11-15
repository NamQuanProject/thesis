# Run the experiments

## 1. Get the code on the GPU server

For a new checkout:

```bash
mkdir -p /workspace/ContinualLearningThesis
git clone https://github.com/NamQuanProject/thesis.git /workspace/ContinualLearningThesis/repository
cd /workspace/ContinualLearningThesis/repository
```

For an existing checkout, use its actual directory and update it before creating
a new run (replace `/path/to/your/thesis-checkout`):

```bash
cd /path/to/your/thesis-checkout
git pull --ff-only origin main
```

## 2. Install the environment

Use Python 3.11 or 3.12 on a CUDA GPU server. Run the setup once per environment:

```bash
bash scripts/setup_server.sh
export THESIS_PYTHON="$PWD/.venv/bin/python"
export THESIS_DATA=/workspace/ContinualLearningThesis/data
export THESIS_RESULTS=/workspace/ContinualLearningThesis/results
```

To reuse the environment from the earlier server setup instead:

```bash
export THESIS_PYTHON=/workspace/ContinualLearningThesis/.venv/bin/python
export THESIS_DATA=/workspace/ContinualLearningThesis/data
export THESIS_RESULTS=/workspace/ContinualLearningThesis/results
```

## 3. Prepare ImageNet-R and pretrained weights

```bash
"$THESIS_PYTHON" scripts/prepare_imagenet_r.py \
  --data-path "$THESIS_DATA" --split-seed 0 --download --weights
"$THESIS_PYTHON" scripts/preflight.py \
  --device cuda --data-path "$THESIS_DATA" --check-data --out preflight.json
```

Reuse the same data directory and `imagenet-r-split-seed0.json` for all compared
runs. To inspect existing data without permitting a download:

```bash
"$THESIS_PYTHON" scripts/prepare_imagenet_r.py --data-path "$THESIS_DATA" --split-seed 0
```

## 4. Create an immutable source checkout and a fresh output directory

```bash
export THESIS_COMMIT="$(git rev-parse --short HEAD)"
export THESIS_RUN_ID="${THESIS_COMMIT}-$(date -u +%Y%m%dT%H%M%SZ)"
export THESIS_SOURCE="/workspace/ContinualLearningThesis/runs/$THESIS_RUN_ID"
export THESIS_OUTPUT="$THESIS_RESULTS/$THESIS_RUN_ID"
git worktree add --detach "$THESIS_SOURCE" HEAD
cd "$THESIS_SOURCE"
```

Keep this checkout unchanged while its jobs run. Use a new run ID for another
attempt; the launcher refuses to overwrite existing results.

## 5. Run the short pilot

Preview the commands first; this does not start training:

```bash
"$THESIS_PYTHON" scripts/run_imagenet_r.py \
  --stage pilot --variants base base_sample class_gauss_sample mass --seeds 15 \
  --data-path "$THESIS_DATA" --output-root "$THESIS_OUTPUT" --device cuda
```

Execute the same pilot:

```bash
"$THESIS_PYTHON" scripts/run_imagenet_r.py \
  --stage pilot --variants base base_sample class_gauss_sample mass --seeds 15 \
  --data-path "$THESIS_DATA" --output-root "$THESIS_OUTPUT" --device cuda --execute
```

## 6. Run the 20-task reference experiment

```bash
"$THESIS_PYTHON" scripts/run_imagenet_r.py \
  --stage compare --variants base --seeds 15 \
  --data-path "$THESIS_DATA" --output-root "$THESIS_OUTPUT" --device cuda --execute
```

The launcher runs jobs sequentially and re-evaluates each saved checkpoint before
starting the next job. Keep the foreground session open until it finishes, or run
the command under your server's configured process supervisor.

## 7. Run the existing per-sample ablations

```bash
"$THESIS_PYTHON" scripts/run_imagenet_r.py \
  --stage compare --variants base_sample class_gauss_sample mass --seeds 15 \
  --data-path "$THESIS_DATA" --output-root "$THESIS_OUTPUT" --device cuda --execute
```

For additional paired seeds, use a new output root:

```bash
"$THESIS_PYTHON" scripts/run_imagenet_r.py \
  --stage compare --variants base base_sample class_gauss_sample mass --seeds 30 45 \
  --data-path "$THESIS_DATA" --output-root "${THESIS_OUTPUT}-additional-seeds" \
  --device cuda --execute
```

To run the existing solver ablations separately:

```bash
"$THESIS_PYTHON" scripts/run_imagenet_r.py \
  --stage compare --variants coupled coupled_raw coupled_gauss --seeds 15 \
  --data-path "$THESIS_DATA" --output-root "${THESIS_OUTPUT}-solver" --device cuda --execute
```

## 8. Optionally enable the faster Jacobian mode

Profile and validate the selected mode on the target GPU:

```bash
"$THESIS_PYTHON" scripts/profile_jacobian.py \
  --data-path "$THESIS_DATA" --out "${THESIS_OUTPUT}-jacobian.json" \
  --required-mode per_sample --required-chunk 0
```

Only after that command succeeds, add these arguments to every compared pilot
and training command, using a fresh output root:

```text
--jacobian-mode per_sample --jacobian-chunk-size 0 --efficient-runtime
```

To impose an optional total launcher time limit, append `--max-seconds 14400`.
An interrupted run cannot resume training automatically; preserve it and use a
fresh output directory for a restart.

## 9. Monitor runs from another terminal

Replace the example path with the run directory printed by the launcher:

```bash
nvidia-smi
cat /workspace/ContinualLearningThesis/results/RUN_ID/compare/base/seed15/progress.json
cat /workspace/ContinualLearningThesis/results/RUN_ID/compare/base/seed15/status.json
tail -n 40 /workspace/ContinualLearningThesis/results/RUN_ID/compare/base/seed15/*.log
ps -eo pid,etimes,args | rg 'main.py|evaluate.py|run_imagenet_r.py'
```

Use the process listing alongside status files after a disconnect or server restart;
a stopped process may leave a stale `running` status on disk.

## 10. Re-evaluate a saved checkpoint

```bash
"$THESIS_PYTHON" evaluate.py --run-dir "$THESIS_OUTPUT/compare/base/seed15"
```

For a completed `mass` checkpoint, run all existing inference modes on the same
images with grouped and mixed batches:

```bash
"$THESIS_PYTHON" scripts/evaluate_specialists.py \
  --run-dir "$THESIS_OUTPUT/compare/mass/seed15" \
  --output-dir "$THESIS_OUTPUT/mass-inference-seed15" \
  --modes anchor soft_mixture full certified budget \
  --protocols grouped mixed --device cuda
```

To check the pilot checkpoint instead, replace `compare/mass/seed15` with
`pilot/mass/seed15` and choose another fresh evaluation output directory.

## 11. Summarize completed results

```bash
"$THESIS_PYTHON" scripts/summarize_results.py "$THESIS_OUTPUT" \
  --csv "$THESIS_OUTPUT/summary.csv"
```

To include pilot records explicitly:

```bash
"$THESIS_PYTHON" scripts/summarize_results.py "$THESIS_OUTPUT/pilot" --include-diagnostics
```

## 12. Run local checks

```bash
"$THESIS_PYTHON" -m pytest -q tests/
"$THESIS_PYTHON" scripts/check_predictive_expansion_math.py \
  --out "$THESIS_OUTPUT/math-checks.json"
"$THESIS_PYTHON" scripts/check_router_capacity_math.py \
  --out "$THESIS_OUTPUT/capacity-checks.json"
```

## 13. Copy artifacts before deleting the server

From a repository checkout on the server, save its Git history with the results:

```bash
git bundle create "$THESIS_RESULTS/source-history.bundle" --all
```

From the local machine, replace `KEY_PATH`, `PORT`, and `HOST` with your SSH values:

```bash
mkdir -p server-backup
scp -i KEY_PATH -P PORT -r \
  root@HOST:/workspace/ContinualLearningThesis/results server-backup/
scp -i KEY_PATH -P PORT -r \
  root@HOST:/workspace/ContinualLearningThesis/runs server-backup/
scp -i KEY_PATH -P PORT -r \
  root@HOST:/workspace/ContinualLearningThesis/logs server-backup/
scp -i KEY_PATH -P PORT \
  root@HOST:/workspace/ContinualLearningThesis/data/imagenet-r-split-seed0.json server-backup/
```

Also copy the `environment.lock.txt` from the directory where setup ran and any
custom launcher or supervisor scripts. Confirm that the transfers succeeded and
that every needed run contains both `backbone.pth` and `evaluation_checkpoint.pth`
before deleting the container.
