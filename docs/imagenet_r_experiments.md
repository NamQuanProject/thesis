# ImageNet-R experiment runbook

This checkout implements HESTIA and Gaussian-routing comparisons. The separate
payback-audit proposal is not present and is not evaluated by these commands.
The target is ImageNet-R, not the full ImageNet-1k training dataset.

## Environment and data preparation

Use Python 3.11 or 3.12. `scripts/setup_server.sh` installs the reference
PyTorch 2.14.1 / torchvision 0.29.1 CUDA 12.6 pair, dependencies, a frozen package
list, an actual tiny IA3/Jacobian device check, and offline tests. The CUDA wheel
must support the rented GPU; the current A100 is supported. It does not install
an NVIDIA host driver or begin training. The pair comes from
[PyTorch's installation matrix](https://pytorch.org/get-started/previous-versions/).

```bash
bash scripts/setup_server.sh
source .venv/bin/activate
python scripts/prepare_imagenet_r.py --data-path /workspace/ContinualLearningThesis/data --download --weights
python scripts/preflight.py --device cuda --check-data --data-path /workspace/ContinualLearningThesis/data
```

If a shared environment was prepared with `uv`, activate that environment instead
and run `uv pip install -r requirements-server.txt`, `uv pip check`, and the same
preflight/tests. Preserve `uv pip freeze` with the run artifacts.

The [ImageNet-R authors](https://github.com/hendrycks/imagenet-r) supply the image
archive. Acquiring data and weights is explicit. Training defaults to
`dtask.download=false`; missing data fails with an actionable message.

For an unsplit archive, the loader records a stratified 80/20 train/test split in
`imagenet-r-split-seed0.json`. Per-class training counts are rounded down, with at
least one example on each side. Split seed 0 is independent of class-order/run
seeds 15, 30, and 45. Original images are never moved. Existing `train/` and `test/`
folders are respected and identified as a different protocol. A partial legacy
split or a changed file inventory causes an error. The inventory identity hashes
relative paths, labels and file sizes; it is not a bytewise image integrity proof.

ImageNet-R does not define this project's training split. These results must be
compared against baselines rerun on the identical saved partition. Do not tune
hyperparameters or select seeds using the final test scores.

## Run sequence

Use an immutable Git commit for each stage. Keep data and results outside the source
checkout. Commands print without executing unless `--execute` is supplied.

```bash
# Two tasks, two training batches/task, two test batches/task. Diagnostic only.
python scripts/run_imagenet_r.py --stage pilot --data-path /workspace/ContinualLearningThesis/data
python scripts/run_imagenet_r.py --stage pilot --data-path /workspace/ContinualLearningThesis/data --execute

# First matched comparison: original GMM routing vs pretrained clean Gaussian keys.
python scripts/run_imagenet_r.py --stage compare --seeds 15 --data-path /workspace/ContinualLearningThesis/data --execute

# Later, predeclared replication and stricter per-sample routing, in a fresh root.
python scripts/run_imagenet_r.py --stage compare --seeds 15 30 45 \
  --variants base class_gauss base_sample class_gauss_sample \
  --data-path /workspace/ContinualLearningThesis/data --output-root results/imagenet-r-replication --execute
```

The default first comparison retains the existing 20 tasks, 200 global class
labels, batch size 16, 25 head updates per batch, and adaptation hyperparameters.
The Gaussian variant changes only the routing key, embedding space, and clean
key view. The published-style protocol routes an entire task-homogeneous test
batch together. Report per-sample results separately; do not pool these protocols.
No result, novelty claim, or accuracy gain is assumed by this runbook.

`--max-seconds` optionally limits the entire launcher, including re-evaluation.
With no limit it continues until completion or an explicit stop. One training
job runs at a time. A failure stops the plan and retains logs; retry in a new
output root. Never silently overwrite failed runs or count duplicate seeds.

## Run records and interpretation

Each training directory contains:

- `config.json`, `provenance.json`, `stream.json`: exact settings, commit/source hash,
  software/device versions, observed task order, and sample identity hashes.
- `backbone.pth`: the exact frozen backbone, saved once for portable evaluation.
- `evaluation_checkpoint.pth`: an atomically written, coherent adapter/head/key
  bundle and accuracy history after each completed task.
- `progress.json`, `status.json`, training logs, and final `result.json`.

`runtime.jacobian_chunk_size=1` computes Jacobian rows sequentially, preserving
the mathematics while reducing peak memory. `runtime.save_ridge_stats=false`
avoids storing large per-segment Gram matrices; solved adapters suffice for
evaluation. Training resume is not implemented. Interrupted runs can re-evaluate
their last complete task but cannot resume optimizer/stream state.

```bash
python evaluate.py --run-dir results/imagenet-r/compare/base/seed15
python scripts/summarize_results.py results/imagenet-r/compare
```

The launcher re-evaluates each completed run and checks agreement before moving
on. Forgetting uses the saved accuracy history; a legacy checkpoint without that
history reports unavailable forgetting. Routing fields in protocol 2 measure
whether a selected adapter's class mask includes the actual labels. This is
**label coverage**, not discovery accuracy or equality of detected segment IDs
with ground-truth task IDs. A merged segment can have high coverage. Report
classification accuracy, forgetting, and the number of detected segments alongside it.

The summarizer excludes diagnostic pilots by default, separates protocol
versions, and rejects duplicate seeds. A pilot checks execution and memory; it
does not estimate full benchmark quality or confirm the research hypothesis.

## Remote operation

Current authorized SSH endpoint: `root@213.5.72.199`, port `31688`.
The user authorized setup and experiments without an elapsed-time cap until an
explicit stop request. They will manage instance deletion. Do not stop/destroy
the rental as a side effect of stopping a training process.

The shared environment is `/workspace/ContinualLearningThesis/.venv`; the Git
clone is `/workspace/ContinualLearningThesis/repository`. Use detached worktrees
for execution and pull new commits into the clone for subsequent runs. Do not
change a running worktree's files. Run long jobs under a named supervisor
program with `autorestart=false`; retries must be recorded, not automatic.

Server processes can survive an SSH disconnect. That is not continuous agent
supervision after this conversation ends. Preserve logs/checkpoints locally
before the instance is deleted: `/workspace` is not necessarily a persistent volume.
