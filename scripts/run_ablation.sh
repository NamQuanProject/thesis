#!/usr/bin/env bash
# Run (or print) the HESTIA improvement ablations for one benchmark config.
#
#   scripts/run_ablation.sh CONFIG "VARIANTS" "SEEDS"
#   scripts/run_ablation.sh configs/imgn-r-seed-15.yml "base cg_pre_clean" "15 30 45"
#   DRY_RUN=1 scripts/run_ablation.sh configs/cifar-100-seed-15.yml all "15 30 45" > jobs.txt   # one command per line
#
# Every run writes results/<config>/<variant>/seed<seed>/result.json;
# summarize with:  python scripts/summarize_results.py results/
# The same seed gives the same data stream for every variant (paired comparison).
set -euo pipefail

CFG="$1"
VARIANTS="${2:-all}"
SEEDS="${3:-15 30 45}"
PY="${PYTHON:-python}"
NAME="$(basename "${CFG%.*}")"

CG_PRE_CLEAN="improve.routing_key=class_gauss improve.routing_space=pretrained improve.key_view=clean"

overrides() {   # variant -> config overrides (bash 3 compatible)
  case "$1" in
    # --- reference ------------------------------------------------------------
    base)                echo "" ;;                                   # original algorithm (this codebase)
    # --- 01: likelihood-correct scoring of the original GMM keys ---------------
    nll)                 echo "improve.gmm_score=nll improve.gmm_var_shrink=10" ;;
    # --- 02: class-conditional order-invariant keys (and where to route) -------
    gmm_pre)             echo "improve.routing_space=pretrained" ;;   # original key, pretrained space
    cg_adapter)          echo "improve.routing_key=class_gauss improve.routing_space=adapter improve.cov_scope=task" ;;
    cg_pre)              echo "improve.routing_key=class_gauss improve.routing_space=pretrained" ;;
    # --- 03: keys from the clean (test-transform) view -------------------------
    cg_pre_clean)        echo "$CG_PRE_CLEAN" ;;
    # --- 04: fuse the class-Gaussian posterior into the routed head ------------
    fusion)              echo "$CG_PRE_CLEAN improve.fusion_alpha=1.0" ;;
    # --- 05: linearized adaptation ---------------------------------------------
    sel_loss)            echo "improve.select_metric=loss improve.select_tol=0.001 improve.refine_max_iters=50" ;;
    joint)               echo "improve.accumulate=joint" ;;
    # --- stricter protocol (report separately: one adapter per test sample) ----
    cg_pre_clean_sample) echo "$CG_PRE_CLEAN improve.routing_unit=sample" ;;
    base_sample)         echo "improve.routing_unit=sample" ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

ORDER="base nll gmm_pre cg_adapter cg_pre cg_pre_clean fusion sel_loss joint cg_pre_clean_sample base_sample"
[[ "$VARIANTS" == "all" ]] && VARIANTS="$ORDER"

for seed in $SEEDS; do
  for v in $VARIANTS; do
    ov="$(overrides "$v")"
    out="results/${NAME}/${v}/seed${seed}/"
    cmd="$PY main.py --cfg $CFG --seed $seed --tag $v --output_dir $out"
    [[ -n "$ov" ]] && cmd="$cmd --set $ov"
    if [[ -n "${DRY_RUN:-}" ]]; then
      echo "$cmd"
    else
      echo ">>> $cmd"
      $cmd
    fi
  done
done
