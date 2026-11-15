"""Collect results/**/result.json into a mean +- std table (markdown).

    python scripts/summarize_results.py results/            # all configs
    python scripts/summarize_results.py results/ --csv out.csv
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = [("avg_acc", "Acc", 1.0), ("forgetting", "Forget", 1.0),
           ("routing_batch", "Coverage(batch)", 100.0), ("routing_sample", "Coverage(sample)", 100.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--csv", default=None)
    ap.add_argument('--include-diagnostics', action='store_true')
    args = ap.parse_args()

    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.root, "**", "result.json"), recursive=True)):
        with open(path) as f:
            r = json.load(f)
        if r.get('kind') == 'diagnostic' and not args.include_diagnostics:
            continue
        # Old index-matching metrics and the new label-coverage protocol are not pooled.
        protocol = r.get('protocol_version', 1)
        kind = r.get('kind', 'benchmark')
        group = (f'{r["cfg"]} / protocol {protocol} / {kind}', r['tag'] or 'base')
        if any(old['seed'] == r['seed'] for old in groups[group]):
            raise ValueError(f'Duplicate seed {r["seed"]} for {group}; summarize one experiment root at a time')
        groups[group].append(r)

    rows = []
    for cfg_name in sorted({k[0] for k in groups}):
        base = groups.get((cfg_name, "base"))
        print(f"\n### {cfg_name}\n")
        print("| variant | seeds | " + " | ".join(m[1] for m in METRICS) + " | dAcc vs base (paired) |")
        print("|---|---|" + "---|" * len(METRICS) + "---|")
        for (c, tag), runs in sorted(groups.items()):
            if c != cfg_name:
                continue
            cells = []
            for key, _, scale in METRICS:
                vals = np.array([r[key] for r in runs if r.get(key) is not None], dtype=float) * scale
                cells.append(f"{vals.mean():.2f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.2f}" if len(vals) else 'n/a')
            delta = ""
            if base and tag != "base":
                b = {r["seed"]: r["avg_acc"] for r in base}
                d = [r["avg_acc"] - b[r["seed"]] for r in runs if r["seed"] in b]
                if d:
                    delta = f"{np.mean(d):+.2f} (n={len(d)})"
            seeds = ",".join(str(r["seed"]) for r in sorted(runs, key=lambda r: r["seed"]))
            print(f"| {tag} | {seeds} | " + " | ".join(cells) + f" | {delta} |")
            rows.append([cfg_name, tag, seeds] + cells + [delta])
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cfg", "variant", "seeds"] + [m[1] for m in METRICS] + ["dAcc_vs_base"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
