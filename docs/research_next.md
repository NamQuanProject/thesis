# Grow What You Can Retrieve

For the current problem-led narrative and 2025-onward gap assessment, see
[When does change require new capacity?](paper_story.md), updated 18 November 2026.

**Scope update, 16 November 2026:** the subsequent request prioritizes capability
acquisition, retention constraints, and bases, with a 2025-onward literature
filter. See [the acquisition and memory review](acquisition_research_2025plus.md)
for the current recommendation and recent baselines outside HESTIA. The routing
proposal below remains a research record, not an implemented growth controller.

## Research decision, 15 November 2026

**Proposed direction:** distinguish routing failure from missing predictive capacity,
then allocate adapters under an explicit retention budget. Use HESTIA as the
reference learner and an initial adapter generator.
Treat adapter creation, selection, and training as decisions about the predictive
distribution actually deployed, rather than three independent improvements.

This is a research blueprint, not an implemented method or a demonstrated
ICML 2027 contribution. The existing `mass` implementation is an earlier inference
prototype. It does not implement the growth rule, retention controller, or
deployment-aware training proposed here. No new training was started for this review.

The candidate contribution is narrower than "utility routing" or "dynamic experts":
**decide whether to repair the router, reuse a predictor, or learn a new expert by
measuring improvement beyond the existing bank's routing capacity, while charging
the old-data risk and resources introduced by that action.** The technical target
is a useful capacity decision under restricted routing information, bounded memory,
and repeated adaptive decisions. Section 8.2 defines an explicit capacity test.

**Decisive novelty correction.** The extended search found
[FireGate, submitted 9 November 2026](https://arxiv.org/abs/2611.11979), which already
protects prior behavior with frozen-feature gates, calibrates them without old
behavior labels, composes protection across a stream, and relates attainable
change to distinguishability. Therefore feature-only retention certification and
its coverage tradeoff cannot be the main novelty claim here. The soft-mixture
envelopes below are useful mathematical components and controls. Merely combining
them with an existing utility router is insufficient for the requested paper.

The proposal is worthwhile only if its capacity and retention bounds can be made
useful at practical memory budgets and the complete policy outperforms simpler,
equally funded controls.

## 1. What the current evidence actually says

The most recent server inspection found the following:

| Run | Observed progress | Macro top-1 accuracy | Interpretation |
|---|---:|---:|---|
| Fork reference `base`, seed 15 | 20/20 tasks, final result saved | 63.0568% | A completed run of this fork; not an exact official-paper reproduction. |
| `coupled`, seed 15 | 18/20 evaluations saved | 40.3112% | Interrupted later during the next evaluation; not a final result. |
| `class_gauss` | Short pilot only | 4.6875% | No completed full benchmark exists in the inspected results. |
| `mass` | Short pilot and paired inference checks only | 43.7500% | Its anchor alone also achieved 43.7500%; this does not establish an adapter gain. |

The two full-run endpoints above have different task counts and must not be
subtracted as a paired final comparison. Previously inspected, matched 15-task
checkpoints gave 64.4833% for `base` and 38.5186% for `coupled`.
The failing `coupled` run uses adapter-space GMM routing, just as `base` does.
It does **not** test class-Gaussian routing. Its modifications include a joint
tangent solve, a rank-128 sketch, and a residual target. Their individual causal
effects have not been isolated.

See [the observation record](run_records/2026-11-15.md) for completion status,
paths, and the interrupted backup. Do not treat absent artifacts as completed runs.

## 2. Repository audit: the three mechanisms

| Mechanism | Current reference implementation | Consequence to investigate |
|---|---|---|
| Growth | `main.py` waits for score convergence and then checks for labels not in the current segment's observed class set. | Same-label domain changes need not create an adapter. A new class can trigger growth without proving additional capacity is useful. |
| Retrieval | `evaluate.py` evaluates candidate adapters, uses GMM scores, selects one per batch by default, and masks classes outside its scope. | Selection errors can assign zero probability to the true class. Grouped and mixed test batches represent different information conditions. |
| Learning | `lib/train.py` trains a shared head, then performs target-probability tangent updates. `main.py` copies the updated head into the Jacobian reference every batch. | The full learning trajectory is not order-invariant merely because its accumulated matrices are additive. Frozen adapters also share a changing head in the reference path. |
| Coupled ablation | `lib/coupled_tangent.py` changes the target, parameter coupling, and history compression. | Better optimization of its surrogate need not improve nonlinear multiclass prediction. A failed combined run does not identify which change failed. |

Additional differences from an official reproduction include the stratified split
manifest, independent loader RNGs, the segment-mask fix, and evaluation bookkeeping.
The paper's appendix reports AdamW and a 5e-4 ImageNet-R head learning rate; this
checkout uses Adam with `HEAD_LR=1e-3`. Record both code and paper settings explicitly.
Changing the data partition or fixing accounting is not a new learning contribution.

## 3. Closest primary work and claims we cannot make

The following is a targeted audit, not an exhaustive priority certification.
Publication venues are stated only where verified; recent preprints remain preprints.

| Primary source | Overlap that matters | Required distinction or control |
|---|---|---|
| [HESTIA, UAI 2026](https://proceedings.mlr.press/v337/le26a.html) | Linearized adaptation, density routing, and a retrieval bound involving estimation quality and separability. | Study predictive value of a growth decision and retention after a new gate is introduced. Do not claim to discover that retrieval matters. |
| [CN-DPM, ICLR 2020](https://arxiv.org/abs/2001.00689) | Task-free expert expansion and probabilistic mixture inference. | Dynamic expert count and soft mixtures are established. |
| [Learn to Grow, ICML 2019](https://proceedings.mlr.press/v97/li19m.html) | Structural reuse, adaptation, and growth. | A reuse/update/new action menu is not new. |
| [AdaNet, ICML 2017](https://proceedings.mlr.press/v70/cortes17a.html) | Complexity-controlled structural growth of an ensemble with learning guarantees. | Residual fitting, reweighting existing predictors, and growing only for improvement are established structural-learning ideas. |
| [SEMA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.pdf) | Representation descriptors trigger adapter expansion; learned composition. | Compare predictive admission against descriptor-based growth under equal budgets. |
| [FiUni, August 2026 preprint](https://arxiv.org/abs/2608.27070) | Fisher-subspace matching guides reuse, expansion, and creation. | Fisher-based growth is not a distinct contribution by itself. |
| [MOS, AAAI 2025](https://arxiv.org/abs/2412.09441) | Adapter merging and self-refined retrieval. | Include an established retrieval/ensemble control. |
| [FDC, September 2026 preprint](https://arxiv.org/abs/2609.34503) | Post-hoc routing calibration using subspaces, residual likelihood, and prototypes. | A better density score or Gaussian calibration alone is insufficient. |
| [Calibrated Knowledge Aggregation, ICML 2026](https://proceedings.mlr.press/v306/mozaffari26a.html) | Utility-trained routing, expert aggregation, and replay in continual VQA. | Directly compare utility routing without the proposed growth/retention controller. Its experts are allocated per training task; our target has no supplied boundaries. |
| [On Token's Dilemma, CVPR 2026](https://arxiv.org/abs/2603.27481) | New experts attract old tokens even with old expert/router parameters frozen; targeted routing regularization. | Include routing-preservation regularization as a control. Neither the interference observation nor protecting old routing is new. |
| [FireGate, November 2026 preprint](https://arxiv.org/abs/2611.11979) | Frozen-feature gating, unlabeled protection calibration, compositional retention, and a distinguishability limit. | A gate-only retention certificate is a direct prior, not our new contribution. Compare against this protection mechanism and identify whether capacity selection adds value. |
| [LEC, ICML 2026](https://proceedings.mlr.press/v306/wang26a.html) | Calibration of selection-conditioned risk, including two-model routing. | Risk-calibrated delegation is established. Our old-population excess risk is a different estimand, not a new general calibration principle. |
| [CP-MoE, August 2026 version](https://arxiv.org/abs/2605.20247) | A transient expert guides routing and protection of stable experts. | A temporary candidate/probe is not new. |
| [Learning to Defer, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html) | Consistent learning of prediction versus delegation. | Conditional-risk routing and its statistical tools require attribution. |
| [MoE Adapters, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.html) | Routing between adapted and original models using an OOD selector. | An OOD fallback is an engineering control, not a new mechanism. |

The paper must survive comparison with utility routing, FireGate-style protection,
and complexity-controlled ensemble growth, both separately and in combination.
"Three improvements to HESTIA" is not enough.
The elementary results below are foundations; presenting their standard convexity
and concentration arguments as entirely new theory would be misleading.

## 4. A common predictive object

Let `P` be the complete incumbent predictive distribution, including all previously
accepted frozen adapters, heads, and gates. Let `q_theta` be a candidate expert.
Both predict over the same declared label space. Let `z = phi_0(x)` be a frozen
router representation, optionally followed by a fixed projection. A candidate gate
`g_psi(z)` lies in `[0, alpha]`, where `alpha < 1`.

The proposed insertion is

\[
 P^+(y\mid x)=(1-g_\psi(z))P(y\mid x)+g_\psi(z)q_\theta(y\mid x).
 \tag{1}
\]

This convex mixture is established machinery, not the novelty claim. Its useful
property is that its current-data gain and its possible old-data damage refer to
the same deployed predictor. Old parameters remain fixed. Freezing parameters
alone does not freeze the mixture's predictions.

For a growing class vocabulary, embed both distributions in a common vocabulary
before comparison and specify the probability-floor rule. Do not use hidden task
IDs, current labels, or test-batch membership to compute the gate. A supervised
label arrives only after the prediction. A temporal router using past labels is
a separate protocol from ordinary independently evaluated test images.

## 5. Adapter growth: benefit after routing, with retention charged

Use an inexpensive prequential-loss trigger to decide when to investigate a change.
A trigger only schedules a probe; it does not prove that a new expert is needed.
Compare three concrete actions under a declared construction budget:

1. Repair routing or reuse frozen experts without adding an adapter.
2. Fit a cheap head-only candidate.
3. Fit a new IA3 adapter and its own head, initially copied from a suitable incumbent.

The first action is essential: improvement over a poor incumbent router is not
evidence that new adapter parameters were needed. Section 8.2 supplies a restricted
convex routing control and a sufficient certificate for benefit beyond that whole
control family. An unconstrained neural router must be measured separately; a
certificate for a small routing class does not rule out its cheaper improvements.

Candidates are trained on a fitting prefix. Freeze each candidate and gate before
its subsequent audit. Audit predictions precede labels; a rejected candidate still
consumes its measured compute and temporary memory. Cap candidate count and fitting
steps. Stop probing when the hard resource budget is exhausted.

For the Brier loss `ell(p,y) = ||p-e_y||^2 / 2`, define current-distribution gain

\[
 G_{\rm cur}=\mathbb E_{Q_{\rm cur}}
 [\ell(P,Y)-\ell(P^+,Y)].
 \tag{2}
\]

This evaluates the actual gate. A perfect expert evaluated with an oracle source
ID is only a diagnostic upper bound and cannot justify admission.

Declare the target evaluation mixture
`Q_star = (1-pi) Q_old + pi Q_cur`. The weight `pi` is an objective choice, not an
oracle probability silently supplied by the benchmark. A source-balanced objective
and a sample-balanced objective must be reported separately.

If `L_cur` is a valid lower bound on (2) and `U_old` bounds the old-data loss
increase, admit only when

\[
 \pi L_{\rm cur}-(1-\pi)U_{\rm old}
   >\lambda_{\rm size}\Delta M,
 \tag{3}
\]

and the physical memory and compute limits are satisfied. `Delta M` includes the
adapter, head, gate, and additional retained statistics. Its scalar price is a
declared tradeoff, not a universal conversion from bytes to accuracy. Prefer the
cheaper admissible action when useful gains are indistinguishable.

For a fixed comparison under conditional iid auditing, `D_i` in (2) lies in
`[-1,1]`. The conservative anytime bound

\[
 L_n=\bar D_n-\sqrt{\frac{2}{n}\log\frac{n(n+1)}{\delta}}
 \tag{4}
\]

follows from Hoeffding and a summable union bound. Allocate error across candidates
and trials. Fitting or repeatedly selecting a candidate on its own audit invalidates
this argument. Arbitrary future drift and changing deployment mixtures are not
covered. A cap is essential: a statistically valid audit can still waste the entire
stream on unsuccessful probes. This standard testing component is not claimed new.

## 6. Retrieval and OOD: predictive routing with bounded interference

### 6.1 What the gate should learn

The gate estimates where a concrete candidate improves the deployed prediction.
It is not supervised with an adapter's birth index. A returned regime can reuse
existing knowledge without requiring a new adapter. Each accepted expert retains
its own head; unrelated later head updates cannot alter that expert's output.

The default gate uses a fixed shared feature space so historical gate statistics
remain meaningful. Test decisions are per sample. Grouped-batch inference remains
a separately reported reference protocol, never pooled with mixed-batch results.

### 6.2 A retention envelope that does not require old images

For any distributions `P,q`, label `y`, and gate `g` in `[0,1)`, (1) implies

\[
 -\log P^+_y+\log P_y\leq-\log(1-g),\qquad P_y>0.
 \tag{5}
\]

Proof: `P^+_y >= (1-g)P_y`. Equality is attained when `q_y=0`.
Thus the log-loss envelope is sharp when nothing about the candidate's old-data
predictions is available. It is not a top-1 accuracy guarantee.

For the normalized Brier loss there is a sharper envelope than the elementary
convexity bound `loss increase <= g`:

\[
 \ell(P^+,y)-\ell(P,y)\leq b(g):=\frac{g}{2-g}.
 \tag{6}
\]

To verify sharpness, the increase is convex in `q`, so its maximum over the simplex
occurs at a vertex. The vertex `q=e_y` cannot increase the loss. For `q=e_j`,
`j != y`, maximizing the resulting concave quadratic in `P` gives
`P_y=1/(2-g)`, `P_j=(1-g)/(2-g)`, with all other coordinates zero. Substitution
gives `g/(2-g)`. This maximizer is feasible, establishing both the bound and its
sharpness. This is an elementary mixture calculation, not an established priority
claim for a new inequality.

Consequently,

\[
 R_{Q_{\rm old}}(P^+)-R_{Q_{\rm old}}(P)
 \leq\mathbb E_{Q_{\rm old}}[b(g(\phi_0(X)))].
 \tag{7}
\]

The right side needs only old frozen features and the new gate. It needs neither
the old labels nor a forward pass of the new adapter on old images. This supplies
a retention constraint for the proposed capacity decision. FireGate already
establishes the broader gate-based protection idea; changing its loss envelope
alone is not the research contribution.

Maintain a bounded feature memory for this purpose. It is **feature replay/storage**,
not replay-free learning. For example, 2,000 float32 768-dimensional features use
6,144,000 bytes before metadata; a fixed 64-dimensional router projection uses
512,000 bytes. Projection reduces storage but may discard routing information.
The representation must remain fixed, or the retained features no longer describe
the deployed gate's inputs.

For a fixed old target distribution `Q`, consecutive accepted insertions satisfy
the deterministic telescoping bound

\[
 R_Q(P_T)-R_Q(P_s)
 \leq\sum_{t=s+1}^T\mathbb E_Q[b(g_t(Z))].
 \tag{8}
\]

A budget therefore must be cumulative for each protected population. Resetting a
small allowable risk after every birth does not establish small lifelong forgetting.
The corresponding log-loss bound sums `E_Q[-log(1-g_t)]`.

### 6.3 Statistics are the difficult part, not the algebra

For a representative iid old-feature sample of size `m` and a **predeclared** gate
class `G`, let `B={b composed with g: g in G}`. A standard uniform bound has the form

\[
 \mathbb E_Q b(g(Z))\leq\widehat{\mathbb E}_m b(g(Z))
 +2\mathfrak R_m(B)
 +b(\alpha)\sqrt{\frac{\log(1/\delta)}{2m}},
 \quad\text{simultaneously for }g\in G,
 \tag{9}
\]

where `R_m` is expected Rademacher complexity. Uniformity, rather than an invalid
fixed-gate Hoeffding test, permits the gate to depend on that memory. Credit the
[standard complexity theory](https://www.jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf).
The feature representation, function class, and its constraints must be fixed for
this statement. Arbitrary data-dependent classes or adaptively selected reservoirs
do not inherit it automatically. An empirical finite-history target needs its own
sampling-without-replacement argument; future old-regime risk needs an additional
sampling assumption. A retained feature is not automatically iid evidence.

Crude bounds may be vacuous. At 20 balanced source groups, current-source weight
can be about 1/20, so even a small old-risk allowance can erase the benefit in (3).
The research must establish a useful localized/low-leakage bound or demonstrate
that the required memory is affordable. Reporting an empirical penalty as a formal
certificate is unacceptable. A controller that never grows has not solved the task.

There is an unavoidable rare-region obstruction. Fix the same candidate, gate,
fitting history, and independent algorithmic randomization in two experiments.
Let the old feature laws be `Q_0=point_mass(z_0)` and
`Q_epsilon=(1-epsilon)*point_mass(z_0)+epsilon*point_mass(z_1)`, with
`g(z_0)=0` and `g(z_1)=alpha`. Choose fixed predictors and labels attaining (6)
at `z_1`; their old Brier risk increases are zero and `epsilon*b(alpha)`.
An iid feature memory from the second law has the same all-`z_0` transcript as
the first law with probability `(1-epsilon)^m`. Consider certification of a risk
budget strictly below `epsilon*b(alpha)`. A procedure that certifies the first
law with probability at least `1-beta`, but falsely certifies the second law with
probability at most `delta`, necessarily needs

\[
 m\geq\frac{\log((1-\beta)/\delta)}{-\log(1-\epsilon)}
 \quad\text{when }1-\beta>\delta.
 \tag{10}
\]

This follows from
`Pr_epsilon(certify) >= (1-epsilon)^m * Pr_0(certify)`.
The claim is a hard-family information bound, not a lower bound for every stream.
For a fixed independently calibrated binary gate and zero observed activations,
the exact binomial upper endpoint `1-delta^(1/m)` exhibits the same rare-event
scaling. Adaptive reuse requires uniform control or independent calibration;
the fixed-gate calculation alone does not solve that problem.

### 6.4 An implementable restricted gate family

Before a general neural gate, test a finite dictionary of `M` fixed regions
`R_1,...,R_M` in the frozen feature space. The entire dictionary must be chosen
independently of the protected iid memory, for example by seeded fixed projections
and predeclared thresholds. Do not build a codebook on that same memory and then
apply a fixed-dictionary argument. This restriction can reduce routing quality;
an unconstrained utility gate remains an empirical control.

For a gate `g(z)=a*1{z in R_j}`, where `0 <= a <= alpha`, let `K_j` count the
memory observations in `R_j`. For `K_j<m`, define `u_j` as the unique solution to
`Pr_{Binomial(m,u_j)}(K <= K_j)=delta/M`; set `u_j=1` for `K_j=m`.
Exact one-sided binomial coverage and a union bound imply, simultaneously over
the dictionary, `Q(R_j) <= u_j` with probability at least `1-delta`.
Hence `U_old=b(a)*u_j` bounds (7). Both the region and its amplitude may be selected
using the memory because the same event covers every region and every amplitude.
This does not authorize enlarging the dictionary adaptively.

For zero observed activations the bound is explicit:

\[
 U_{\rm old}=b(a)\left[1-(\delta/M)^{1/m}\right].
\]

With `a=0.8`, `M=256`, `delta=0.01`, and `m=2000`, this is approximately 0.00338
in normalized Brier units. If `pi=1/20` and the size penalty is zero, condition
(3) requires `L_cur > 19*U_old`, approximately 0.0642. This is an arithmetic
feasibility illustration **conditional on zero activations and the sampling
assumptions**, not a measured routing result or a promise that the gate has power.
It also does not remove uncertainty from the current-data audit.

Reusing the same valid event for later selections from this fixed dictionary is
allowed, but cumulative old-risk spending still applies. Replacing the memory,
changing the protected population, or adding a learned dictionary requires a
new valid argument and error accounting. A current stream of nonidentical regimes
does not automatically produce an iid sample from every protected distribution.
The confidence construction is classical; useful adaptive gate/memory allocation
beyond this restricted control is the open contribution, not binomial inversion.

### 6.5 Coverage OOD is a separate diagnostic

Use a simple frozen-space coverage score, such as a declared nearest-neighbor
distance, as a control for inputs unsupported by retained data. Calibrate thresholds
on permitted training/calibration observations, not future test classes. Return a
fallback prediction and an unsupported-input flag when appropriate. The ViT's
frozen representation is not itself a trained zero-shot classifier for new labels.

A low utility gate does not imply OOD: the incumbent may already be correct. A high
loss does not imply useful capacity: the label may be noisy. A feature shift does
not imply changed predictive requirements. If `P(X)` is identical but `P(Y|X)`
changes, current features alone cannot identify the hidden regime; prior revealed
labels can help only in an explicitly temporal evaluation protocol.

## 7. How each adapter should learn

Start with the same frozen ViT and IA3 parameterization as the reference. Use an
ordinary, bounded Adam/SGD training budget for the new adapter and its own head.
Compare this directly with the original block tangent learner before introducing
LoRA or a new optimizer. More optimization sophistication is not presumed better.

Train the predictor that will actually be deployed:

\[
 \mathcal L(\theta,\psi)=
 \widehat{\mathbb E}_{\rm fit}[\ell(P^+,Y)]
 +\lambda_{\rm ret}\widehat{\mathbb E}_{\rm memory}b(g_\psi(Z))
 +\lambda_\theta\|\theta-\theta_{\rm init}\|^2.
 \tag{11}
\]

For the primary capacity study, use normalized Brier loss consistently in fitting,
the convex routing control, and admission. Include NLL fitting as a performance
control; gains for one loss do not imply gains for another.
The empirical retention term guides optimization; the separate valid bound decides
whether a formal budget was met. The old predictor is stop-gradient and frozen.
Use full declared-class normalization, not a current-batch class mask that makes
the deployment loss disappear on difficult competing classes.

For Brier fitting, the candidate gradient is
`g*(P_plus-e_y)^T * derivative_theta(q)`, and the gate derivative is
`<P_plus-e_y, q-P>`. These gradients fit the residual prediction error after
the old bank's routing repair. They are elementary derivatives.

For the NLL variant and a fixed gate, the exact candidate gradient is

\[
 \nabla_\theta[-\log P^+_y]
 =r(x,y)\nabla_\theta[-\log q_{\theta,y}],\quad
 r=\frac{gq_{\theta,y}}{(1-g)P_y+gq_{\theta,y}}.
 \tag{12}
\]

The gate derivative is `(P_y-q_y)/P^+_y`. These are standard mixture responsibilities,
not new EM theory. Their role here is to align candidate learning with the exact
insertion audited by (2) and constrained by (7). Pretraining the candidate for a
short declared warm start avoids starving it when `g` starts near zero. All such
steps and rejected candidates are charged. This learning procedure is order
sensitive; do not inherit HESTIA's order-invariance claim.

Use Brier loss for the bounded admission proof and measure both Brier and NLL.
Improving either does not prove improved accuracy. The NLL variant must report
its mismatch with the Brier-based capacity test rather than transferring that
test's guarantee to log loss.

Repeated insertions give explicit stick-breaking weights
`w_0=product_t(1-g_t)` and `w_j=g_j*product_{k>j}(1-g_k)`.
Gates can be computed cheaply from one frozen representation, but evaluating every
expert still costs `O(K)` forwards. A top-k truncation changes the predictor and
must be audited as that predictor. If omitted mixture weight is `u`, completing
with any probability vector changes the full mixture by at most `2u` in L1.
Lazy inference is an efficiency extension, not a new-growth result.

## 8. Why accessible information limits useful specialization

### 8.1 Information available to the router

Let `V` denote information available to a diagnostic oracle and `Z` the information
actually available to a hard router, with `Z` measurable from `V`. For two fixed
predictors, define `d(V)=E[loss(P,Y)-loss(q,Y)|V]`. The optimal hard-switch gains are

\[
 G_V=\mathbb E[d(V)_+],\qquad
 G_Z=\mathbb E[(\mathbb E[d(V)\mid Z])_+].
\]

Then

\[
 G_V-G_Z=\tfrac12\left(\mathbb E|d(V)|-
 \mathbb E|\mathbb E[d(V)\mid Z]|\right)\geq0.
 \tag{13}
\]

Proof: use `a_+=(a+|a|)/2`, the tower property, and conditional Jensen. This is a
standard value-of-information identity, included to define an interpretable
diagnostic rather than advertised as a new general theorem. The oracle might know
the latent regime; it must not be used for deployment. An oracle using the current
label is an even stronger hindsight diagnostic and must be named accordingly.

For soft mixing and Brier loss, put `v=q-P`,
`a(Z)=E[<e_Y-P,v>|Z]`, and `c(Z)=E[||v||^2|Z]`. Its conditional improvement is
`g*a - g^2*c/2`, maximized by `clip(a/c,0,alpha)` when `c>0`; if `c=0`, there is no
candidate change. Thus a soft mixture can help even where hard switching cannot.
Do not apply the hard-switch formula (13) as an upper bound on all soft mixtures.

### 8.2 A candidate must beat router repair, not just the deployed router

Fix a population `Q`, a frozen expert bank `E`, and a declared router family `G`.
Let `C(E,G)` be its attainable predictive distributions and define

\[
 V(E,G)=\inf_{F\in C(E,G)} R_Q(F),\qquad
 \Delta_{\rm capacity}=V(E,G)-V(E\cup\{q\},G).
\]

The deployed risk separates exactly into a routing/optimization deficit
`R_Q(P)-V(E,G)` and the risk attainable by the bank. An improvement over `P`
can remove the first term without providing any new capacity. Keeping `G` the same
on both sides is necessary: a larger router can otherwise masquerade as a better
expert. This decomposition defines the research object, not a new algebraic theorem.

A practical restricted control partitions frozen features into `J` fixed cells.
Within each cell the router uses arbitrary convex weights over `K` frozen experts.
The resulting class `C` is convex; fitting its weights with Brier loss is a convex
quadratic problem on cached expert predictions. It needs neither a task ID nor
per-example oracle expert selection. Every expert forward and cached probability
is charged. A general neural gate is a stronger control outside this certificate.

For any `P in C`, define the population first-order gap

\[
 H_Q(P)=\sup_{F\in C}\mathbb E_Q
 \langle e_Y-P(X),F(X)-P(X)\rangle.
 \tag{14}
\]

Convexity of Brier risk gives
`0 <= R_Q(P)-V(E,G) <= H_Q(P)`. This is the standard conditional-gradient dual-gap
argument, not a new optimization theorem; see
[Jaggi, ICML 2013](https://proceedings.mlr.press/v28/jaggi13.html).
For a new insertion `P_plus` with current-population gain `G_new`, it follows that

\[
 \Delta_{\rm capacity}\geq V(E,G)-R_Q(P^+)
 \geq G_{\rm new}-H_Q(P).
 \tag{15}
\]

The first inequality requires `P_plus in C(E union {q},G)`: keep the same cells
and allow a cellwise insertion weight. If a lower confidence bound on `G_new`
exceeds an upper bound on `H_Q(P)` plus the declared complexity price, this
certifies improvement beyond **every** reweighting of the old bank in that router
class, not just beyond its current imperfect weights. Reusing an existing expert
cannot pass a valid positive-capacity certificate under these same conditions.

For computation, the linear supremum in (14) selects one expert per cell, so its
empirical value is the sum of per-cell maxima of residual-alignment sums divided
by sample count. It can be computed in `O(nK)` after the forwards, despite there
being `K^J` extreme routers. For fresh iid audit data with predictors and cells
fixed before auditing, each alignment is in `[-2,2]`. Hoeffding plus a union bound
therefore gives the conservative fixed-time bound

\[
 H_Q(P)\leq \widehat H_n(P)+
 \sqrt{\frac{8}{n}\left(J\log K+\log(1/\delta)\right)}.
 \tag{16}
\]

Combine it with the gain bound using separate error allocations. The estimate
on the same data used to fit `P` is not a fresh-audit certificate. For repeated
looks or adaptive trials, allocate error and preserve the conditional sampling
assumptions. A bound on `Q_cur` establishes capacity on `Q_cur`; it is not an
old-regime guarantee. Apply the separate retention constraint for that purpose.

This basic bound will often be too loose. A useful research result must exploit
localized disagreement or variance and account for router refinement, retained
memory, and repeated proposals. If all we obtain is (15) plus an ordinary union
bound, the mathematical contribution is too incremental. The purpose of the
restricted control is to expose the missing technical work and provide a precise
first falsification experiment.

### 8.3 Development target

Compare three uses of a fixed incremental budget: repair/refine the router, improve
an existing expert, or add an expert that supplies residual predictive capacity.
Develop a sequential allocation rule whose guarantee remains useful after paying
for those comparisons and protecting old populations. Candidate learning should
reduce the residual of the best fitted old-bank mixture, not merely distinguish
new labels from old labels. The mixture loss in (11), initialized from that fitted
control, is the first implementable training objective.

The requested paper requires a new, nonvacuous allocation/identification result
and evidence that this decision changes the accuracy-resource frontier. Neither
the standard projection geometry nor the feature-only retention bound establishes
that result. Novelty of the complete proposal remains open after this review.

## 9. Concrete implementation sequence

Keep the reference path immutable. New modules should accept generic `predict(x)`
expert callbacks and a fixed router embedding; IA3 is the first implementation,
not an assumption of the theory.

| Stage | Work | Required evidence before proceeding |
|---|---|---|
| 0 | Audit saved reference/coupled checkpoints, same source groups and masks. Measure true-label-admissible oracle head accuracy, routed accuracy, mask exclusions, and shared-head versus frozen-head effects. | Identify whether most recoverable loss is in routing, expert predictions, or their interaction. No inferred cause from the combined coupled run. |
| 1 | Freeze the current adapter bank; fit cellwise convex routing and a small neural utility gate as established controls. Compare hard routing, conventional soft mixing, and FDC-style calibration where compatible. | Measure the routing deficit and whether the first-order capacity gap is estimable before assigning gains to growth. |
| 2 | Implement the generic insertion operator, gate-feature memory, cumulative retention ledger, and reported bounds. Verify (5)-(8) on adversarial simplex fixtures. | Memory accounting, valid statistical assumptions, and bounds that can sometimes admit useful changes. |
| 3 | Compare block-tangent training with ordinary IA3/head training under matched examples and compute. Add objective (11) as a separate ablation. | Gains survive a strong ordinary optimizer; no simultaneous solver/router change. |
| 4 | Add fitting/audit separation and reuse/head/new candidate admission from (3), without supplied task boundaries. | Full costs include unsuccessful probes; a no-growth policy and a fixed-schedule policy are controls. |
| 5 | Repeat with LoRA or a feature-level expert implementation. | The retention interface works independently of HESTIA and IA3. |

The proposed trainer/controller is not yet runnable. Existing run commands in
`README.md` execute only the reference and explicitly named existing ablations.
Do not create a CLI variant whose name implies this proposal has been implemented.

## 10. Experiments needed for a substantial paper

Start with ImageNet-R, 200 classes and 20 construction groups. Use an immutable split
manifest shared by every compared method. Report both the fork reference protocol
and any verified official reproduction; never pool their scores. Use the same
initial weights, class order, data access, and seed per paired comparison.

Predeclare a small development configuration before running the full seed set
15/30/45. Select settings using training-side development data only. One final
seed and tiny pilots are not enough to establish superiority.

The study should contain:

- Standard disjoint-class ImageNet-R for continuity with the reference.
- Mixed-source test batches and single-image decisions, reported separately from
  grouped-batch routing.
- Recurring classes and same-label style shifts, so growth is not equivalent to
  detecting newly seen labels.
- Harmless nuisance changes, irreducible label noise, and controlled hidden-regime
  ambiguity. Synthetic label reversal tests identification, not natural-image SOTA.
- Two additional image streams or backbones to test generality after viability.
- Retained-feature budgets, adapter-count budgets, total fitting compute, rejected
  probes, and per-image expert evaluations. Give the reference the same feature
  memory through a simple calibration/replay control; comparison against a zero-memory
  reference alone cannot establish the new controller's value.

Required ablations are a fixed expert schedule, utility routing alone, preservation
regularization alone, routed-gain admission without retention, oracle-gain admission,
the complete rule, and a cheap head-only/ordinary-update learner. Reuse all methods'
permitted fitting data fairly; audit withholding and extra passes must be disclosed.

Report final macro accuracy, average incremental accuracy, per-source forgetting,
NLL, Brier, coverage versus error, AUROC/FPR95 for the separate OOD protocol,
accepted/rejected probes, realized old-risk changes, bound violations under their
stated assumptions, total wall time, and peak/persistent bytes. Frozen expert
accuracy and deployed accuracy must be separate diagnostics.

**Rejection criteria:** the full method fails to beat the utility-router control
at matched budgets; gains vanish when the reference receives equivalent memory;
the retention bound is usually vacuous; the controller rejects all meaningful
growth; or apparent gains require oracle test grouping. In those cases, narrow or
redirect the paper rather than add more modules.

## 11. Contribution and claim ledger

| Claim | Status |
|---|---|
| Dynamic experts, utility routing, mixture responsibilities, OOD fallback | Established; not our novelty. |
| Pointwise and cumulative retention envelopes | Derived above; require independent mathematical review before publication. No priority claim. |
| Fixed-gate hidden-region memory obstruction | Derived for the stated family; not a universal lower bound or an adaptive-controller theorem. |
| Gate-based protection calibrated without behavior labels | Direct prior in FireGate; not a new research claim here. |
| Brier residual/dual-gap capacity test | Derived application of standard convex optimization; not sufficient novelty alone. |
| Useful sequential decisions between router repair and new expert capacity under memory/retention constraints | Central proposed contribution; not established, implemented, or proven optimal. |
| Better ImageNet-R performance than HESTIA | Unproven. The coupled result is negative evidence; mass has only pilot evidence. |
| ICML 2027 readiness | Requires a distinct result, nonvacuous theory, a complete implementation, and strong matched-budget experiments. No acceptance prediction is justified. |

The most promising story to test is **grow only beyond what the existing bank's
router can recover, while charging retention and resource costs**.
The next implementation should test that story as cheaply as possible before
funding another full grid.

## 12. Checks performed for this proposal

Two NumPy-only scripts were executed on 2026-11-15:

- `scripts/check_predictive_expansion_math.py`: 10,000 random simplex cases and
  an explicit extremal family check the Brier/log envelopes, soft-mixture gain,
  and the NLL gate derivative. Results:
  [retained JSON](run_records/2026-11-15-math-checks.json).
- `scripts/check_router_capacity_math.py`: 1,000 finite-population comparisons
  check the dual-gap and capacity inequalities. A duplicate expert improves a
  poor router but does not yield a positive capacity lower bound; a candidate
  beyond the old convex hull does in a separate fixture. Results:
  [retained JSON](run_records/2026-11-15-capacity-checks.json).

All checks passed. They verify arithmetic and diagnostic distinctions, not proofs,
statistical coverage, novelty, or performance on learned image models. No independent
review of these new derivations is claimed. The four-variant pilot command in
`README.md` was also checked in print-only mode, with no run directory created.
The training algorithm and saved running-experiment sources were not changed.
