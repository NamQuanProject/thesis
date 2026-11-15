# Research candidate: coupled residual tangent experts

Status, 2026-11-15: implemented and covered by small mathematical/integration
tests; the inspected server comparison is negative. At matched 15-task
checkpoints, the fork reference reached 64.4833% and this coupled configuration
38.5186%. The coupled run did not finish all 20 tasks. See the
[run record](run_records/2026-11-15.md); the joint solve, sketch, and target changes
have not been isolated. This is not an established improvement or an ICML-ready
paper. The [subsequent research review](acquisition_research_2025plus.md) studies
acquisition and retention under limited historical information.

Working paper question: **When do independently fitted adapter updates interfere,
and can a bounded-memory joint residual fit improve task-free continual learning?**

## Method and distinction from the existing fork

The released training path computes a target-class probability Jacobian for
each IA3 parameter block. It solves each block against the same target, then
applies all the block updates simultaneously. It accumulates blockwise normal
equations and normally merges their local solutions. The existing Gaussian-key
alternative changes retrieval; it does not address this adaptation objective.

Let the frozen-reference target probability on a batch be p, the chosen soft
target be z, and let A = [A_1 ... A_L] concatenate all IA3 Jacobian blocks. The
first-order probability is p + A delta. A joint residual fit therefore solves

    min_delta ||A delta - (z-p)||^2 + lambda ||delta||^2,
    delta = A^T (A A^T + lambda I)^(-1) (z-p).

The candidate uses this joint fit in the existing training-batch refinement.
It retains the same head updates, refinement selection criterion, detector,
training observations, and class masks. To consolidate batches without a dense
P-by-P matrix, it retains S = R A_history and q = R r_history, where R has
independent Gaussian entries of variance 1/s and s is a declared sketch rank.
Each new batch contributes columns of R to the additive summaries. The deployed
update is S^T (S S^T + lambda I)^(-1) q. Rank is fixed at 128 in the first study.
No raw images or per-example stored gradients are replayed. State costs O(sP+s),
and computing S S^T costs O(s^2 P); this cost must be included in timing.

The sketch has a private RNG. Historical rows were computed with historical
heads, as in HESTIA. Consequently, this is a joint fit to those historical local
surrogates, not an exact relinearization of all past examples under the latest
head. Full training is not claimed to be order invariant.

## Mathematical checks

These are elementary linear-algebra results, not new general theorems.

**Cross-block interference.** For a fixed design A and response r, define
F(w)=||Aw-r||^2+lambda||w||^2, H=A^T A+lambda I,
D=blockdiag(A_l^T A_l+lambda I), E=H-D, g=A^T r.
Let w_ind=D^(-1)g and w_joint=H^(-1)g. Since H is positive definite,

    w_ind - w_joint = H^(-1) E w_ind,
    F(w_ind)-F(w_joint) = w_ind^T E H^(-1) E w_ind >= 0.

Proof: H w_ind = g+E w_ind; subtract H w_joint=g and complete
the square in F. This concerns one fixed ridge objective, not nonlinear test
accuracy, HESTIA's entire accumulation rule, or population forgetting.

For A=[1,1] and r=1, the independent solution is
(1/(1+lambda),1/(1+lambda)), whereas the joint solution is
(1/(2+lambda),1/(2+lambda)). As lambda tends to zero, independent
updates predict an increment of 2 instead of 1. Repeated blocks give an
arbitrarily large amplification example. Actual trained ViT blocks need not
have this geometry; the experiment must determine its relevance.

**Residual target.** In the same linearization, fitting A delta to z rather than
z-p changes the predicted output to p+z when the fit is exact. Subtracting p
corrects the intercept for this surrogate. It does not ensure the nonlinear
network attains that output or improves generalization.

**Sketch scope.** Suppose, for a fixed A,r, R satisfies
(1-epsilon)||Av-r||^2 <= ||R(Av-r)||^2 <= (1+epsilon)||Av-r||^2
for every v. Let v_hat minimize the sketched ridge objective and v_star
minimize F. Then

    F(v_hat) <= ((1+epsilon)/(1-epsilon)) F(v_star).

Proof: the same sandwich bounds hold for the objectives after adding the
nonnegative ridge penalty. Apply the lower sandwich at v_hat, optimality
of v_hat in the sketched objective, and the upper sandwich at v_star.
This is a conditional deterministic statement. We have not proved that rank
128 gives this property for adaptively generated training Jacobians. Standard
fixed-design sketch guarantees cannot silently be applied to that adaptive
case. The uncompressed optimum can be better than the sketched solution.

Additivity gives order invariance only if rows, responses, and their assigned
sketch columns are fixed before reordering. The complete learner changes heads,
soft-target selections, and detection boundaries over time; it has no such
unconditional invariance guarantee.

## Closest work and the unresolved novelty question

- [HESTIA](https://github.com/Alisia0303/HESTIA) supplies the task-free IA3
  pipeline and linearized-adaptation baseline. The candidate modifies its
  actual fitting objective and cross-block coupling.
- [Heckel, AISTATS 2022](https://proceedings.mlr.press/v151/heckel22a.html)
  already studies continual learning with sketched Jacobians and the limitations
  of curvature approximations. Sketching Jacobians to retain interference
  information is therefore not a defensible first-use claim.
- [Scaling Neural Tangent Kernels via Sketching and Random Features](https://proceedings.neurips.cc/paper_files/paper/2021/file/08ae6a26b7cb089ea588e94aed36bd15-Paper.pdf)
  establishes prior work on scalable tangent kernels; ordinary joint ridge
  fitting and sketching are credited tools.
- [Kernel Continual Learning](https://proceedings.mlr.press/v139/derakhshani21a.html)
  uses kernel ridge classifiers with an episodic coreset. Our current derivative
  summaries have a different interface; that difference alone does not establish
  sufficient novelty.

The possible contribution is the identification and treatment of interacting
IA3 updates, intercept error, and streaming consolidation in a task-free adapter
bank, supported by a convincing accuracy/resource improvement. Priority and
significance of that combination remain open. Merely correcting an equation or
combining established tools may be insufficient for a new conference paper.

## Predeclared experiments and falsification

Keep the existing seed-15 HESTIA and Gaussian-routing run as controls. On the
same ImageNet-R partition, class order, batch size 16, and head/refinement
settings, run these additional variants:

| Variant | Adaptation | Routing | Purpose |
| --- | --- | --- | --- |
| coupled | joint sketched residual fit | original GMM | candidate |
| coupled_raw | joint sketched fit without subtracting p | original GMM | residual-offset ablation |
| coupled_gauss | joint sketched residual fit | pretrained clean Gaussian | interaction with existing routing alternative |

Report final average accuracy, forgetting, detected segments, label coverage,
elapsed time, peak memory, and failures. Partial-task scores and diagnostic
pilots are not final results. Use at least three prespecified seeds before
making a reliable performance claim. Do not select rank, ridge, or refinement
settings from final test scores; further development requires a training-only
validation split. Additional ablations must separate sketch error from coupling,
and compare a residual correction of the independent solver.

Reject or revise the hypothesis if coupled adaptation fails to improve the
matched controls, introduces instability, or owes an apparent gain only to
routing changes, extra compute, a favorable seed, or the split protocol.
ImageNet-R is an initial feasibility study, not sufficient evidence of general
continual-learning superiority.

## Server commands

Use the shared server environment and an immutable checkout containing this
implementation. The original running checkout must not be edited.

```bash
python scripts/profile_jacobian.py --data-path /workspace/ContinualLearningThesis/data --out jacobian-profile.json
python scripts/run_imagenet_r.py --stage pilot --variants coupled coupled_raw coupled_gauss \
  --jacobian-mode per_sample --jacobian-chunk-size 0 --efficient-runtime \
  --data-path /workspace/ContinualLearningThesis/data --output-root results/coupled-study --execute
python scripts/run_imagenet_r.py --stage compare --seeds 15 --variants coupled coupled_raw coupled_gauss \
  --jacobian-mode per_sample --jacobian-chunk-size 0 --efficient-runtime \
  --data-path /workspace/ContinualLearningThesis/data --output-root results/coupled-study --execute
```

The per-sample Jacobian implementation and reduced logging/barriers are
engineering changes. CPU double-precision Jacobian agreement, exact head and
optimizer agreement, and GPU training-batch agreement are checked separately;
they are not the paper contribution. GPU throughput profiles collected while
another job runs must disclose that contention.
