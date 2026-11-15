# Learning new capabilities under a retention and memory budget

Research review: 16 November 2026. Status: proposal, not an implemented learner.

This review follows the request to investigate capability acquisition before
starting another server experiment. The literature considered for this update
is from 2025 onward, including nearby work on plasticity, compressed gradient
memory, and certified adaptation. Earlier research notes remain historical
records; their older references are not the baseline shortlist for this proposal.
Repository documentation is in English. The README remains a run guide.

## 1. Recommended scientific question

**Can a continual learner distinguish insufficient capacity from insufficient
information about whether an update will damage earlier predictions?**

The practical objective is to learn a new predictive capability while satisfying
an explicit retention contract and a total memory budget. Task-free operation is
an additional setting, not the main novelty claim. HESTIA supplies one adapter
implementation and comparison; the question also applies to continual LoRA and
other parameterizations.

A high loss is a reason to investigate. It does not establish distribution shift,
reducible error, insufficient model capacity, or the usefulness of another expert.
Even with a stable distribution, noise, calibration, optimization, or routing can
produce high loss. Conversely, distribution change can leave prediction quality
unchanged.

The proposed distinction is between:

1. **A certified feasible update:** existing parameters can achieve a declared
   acquisition target while respecting the retention contract.
2. **Certified insufficiency within a declared model class:** even an optimistic
   interpretation of the retained historical information cannot achieve that
   target under the contract.
3. **Insufficient evidence:** the answer depends on historical directions that
   were discarded. This is not evidence that another adapter is necessary.

Creating a module is a subsequent decision. It must demonstrate useful new
prediction and satisfy its own deployment-level retention check. An impossibility
result for one model class does not establish that a larger model will succeed.

The promising research target is a useful acquisition decision with a quantified
cost of lost historical information. The elementary inequalities below are not,
on their own, an ICML contribution. Priority and practical value remain open.

## 2. Recent work that changes the proposal

These are primary sources. A preprint, a workshop paper, and a main-conference
paper are identified separately. Reported findings are the authors' claims;
their ImageNet-R performance has not been reproduced in this repository.

| Work | Relevant mechanism | What this rules out as our main novelty |
| --- | --- | --- |
| [SD-LoRA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/92f43b1d33fae4aa1958f75317f0cec1-Abstract-Conference.html), [January 2025 preprint](https://arxiv.org/abs/2501.13198) | Keeps earlier low-rank directions while learning magnitudes and new directions; the final model supports direct inference without selecting a task component. | Reusing directions, separating magnitude from direction, and avoiding test-time expert selection are established. This is the primary baseline outside HESTIA. |
| [CL-LoRA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf), [May 2025 preprint](https://arxiv.org/abs/2505.24816) | Shared and task-specific adapters, orthogonal constructions, distillation, and learned block weights. | Shared/private adapter decomposition and preserving common knowledge are insufficient novelty claims. |
| [Keep moving, CoLLAs 2025](https://proceedings.mlr.press/v274/anthes25a.html) | Distinguishes activation changes visible to a task readout from changes in its null space. | Protecting all feature variation can unnecessarily reduce plasticity; output-relevant protection already has direct precedent. |
| [QGPM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/24e8b46430df965674221665816a4964-Abstract-Conference.html) | Quantizes historical subspaces, adjusts gradient projection for quantization error, and uses sparse sketching. | Accounting for compression error in continual projection is already studied. A memory-aware projection alone is not enough. |
| [DYRA, ICML 2026 workshop](https://openreview.net/pdf/cb68cae0fe3904ee13d7cf5ca49aec01bed27141.pdf) | Uses uncaptured activation spectra in interference bounds and selects adapter ranks with a spectral threshold. | Dynamic rank with a spectral-tail forgetting argument is already a close method. This is a workshop paper, not an ICML main-track acceptance. |
| [BISECT, November 2026 preprint](https://arxiv.org/abs/2611.03574) | Splits a rank budget into shared and novelty directions and studies reduced plasticity from restrictive basis selection. | A generalized-eigenvector basis or shared/novel split is not a sufficient new contribution. |
| [FiUni, August 2026 preprint](https://arxiv.org/abs/2608.27070) | Fisher-subspace similarity selects reuse, expansion, or a new module. | Subspace-based grow/update decisions in task-free learning already exist. |
| [Interference and Retention in Continual Learning, July 2026 preprint](https://arxiv.org/abs/2607.09202) | Studies interference geometry, shared-model distortion, capacity, and allocation; includes covariance sketches and compression-related retention arguments. | An old-risk quadratic, a capacity-versus-control decomposition, or a covariance sketch cannot be presented as new by itself. |
| [Predicting Plasticity, May 2026 preprint](https://arxiv.org/abs/2605.09044) | Gives counterexamples to rank-based trainability diagnostics and studies a gradient-based optimization-gain measure. | Feature rank alone is not a reliable certificate of learnability. |
| [The Dual Nature of Plasticity Loss, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6489f2c6ac6420124fcef2a489615a97-Abstract-Conference.html) | Separates inability to fit from deterioration in generalization. | Training improvement and acquisition of a useful capability must be measured separately. |
| [Pi-CCA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e4165c96702bac5f4962b70f3cf2f136-Abstract-Conference.html) | Retains a compact summary of cross-modal alignment for continual adaptation. | A compact geometry summary used to preserve earlier knowledge is established in an adjacent problem. |
| [CerCE, public ICLR 2026 submission](https://openreview.net/pdf/9d1cc2832130c15e2b4c8292bbb7d1aab62625e8.pdf) | Proposes constrained continual updates using output certificates derived from weight-perturbation bounds. | Certified retention and constrained adaptation are not first-use claims. Acceptance status was not verified from the primary forum. |

The closest collision is the combination of QGPM, DYRA, and the interference
geometry preprint. The inspected material already addresses compressed geometry
and capacity. The narrower object proposed here is a **two-sided, computable
answer to acquisition feasibility given an incomplete historical state**, with
an explicit unresolved region. This is a candidate distinction, not an assertion
that no previous paper has studied it.

### Reading scope

The SD-LoRA, CL-LoRA, and QGPM primary PDFs were inspected. The FiUni and BISECT
methods and relevant interference-geometry sections were inspected in primary
HTML. QGPM's Sections 3.3--3.4 specifically cover error-aware projection and
sparse embedding; neither a Gaussian compression-error model nor an embedding
bound should be silently replaced by a deterministic covariance certificate.

DYRA's indexed primary PDF supplied its method and rank-allocation derivation;
direct PDF retrieval was blocked. CerCE was checked through indexed primary
PDF passages, with the forum blocked. Pi-CCA and the plasticity papers inform
scope from their primary abstracts; this is not a full proof audit of them.
No priority conclusion relies on an aggregator's summary or review.

## 3. What the current code and failed run actually imply

In this checkout, IA3 is the adaptation mechanism and the GMM is a detection and
retrieval mechanism. They are different objects.

- `main.py` updates the classification head and copies it into the reference used
  to compute subsequent Jacobians. Consequently, the full learner is not order
  invariant just because it sums normal-equation statistics.
- The reference path shares a changing head. Preserving an IA3 vector alone does
  not preserve the old predictor, much less its deployed routing decision.
- `lib/coupled_tangent.py` stores a Gaussian row sketch. Its state does not include
  the deterministic one-sided covariance-error bound assumed in Section 4.
  The new proposal cannot treat that existing sketch as certified memory.
- The coupled ablation changed the joint solve, target offset, and compression
  together. A better value of its local least-squares surrogate need not produce
  better nonlinear multiclass prediction.

The matched 15-task observation was 64.4833% for the fork reference and 38.5186%
for the coupled configuration. Both used adapter-space GMM routing. This is a
negative result for that configuration, not evidence against all coupled solves,
and not a comparison of GMM versus class-Gaussian retrieval. See the
[actual run record](run_records/2026-11-15.md).

Before another method comparison, separate surrogate fit, actual prediction
change, old/new classification, and retrieval. Changing all four at once would
repeat the attribution problem.

## 4. A tractable mathematical foundation

### 4.1 Start with an explicit retention contract

Consider an affine predictor in fixed coordinates:

$$f_w(x)=b(x)+J(x)w.$$

The vector output includes every class relevant to the prediction contract.
The functions `b` and `J` are fixed. This describes an exact linear readout or
an explicitly deployed tangent model. It is only an approximation to a nonlinear
IA3 network evaluated at updated weights.

Let `w_ref` be a saved reference and let historical observations be indexed by
`i=1,...,n`. Define the mean squared change in their output vectors:

$$D(w)=\frac1n\sum_i\|f_w(x_i)-f_{w_{ref}}(x_i)\|_2^2
       =(w-w_{ref})^T G(w-w_{ref}),\qquad
  G=\frac1n\sum_iJ(x_i)^TJ(x_i).$$

Require `D(w) <= epsilon`. This is a prediction-preservation contract on a stated
historical population. It is not a claim that all retained predictions are correct.
It is not itself a population-accuracy certificate.

For several protected cohorts, retain a separate reference and tolerance for
each and impose all constraints. A single pooled average can conceal damage to
a small cohort. References and budgets persist across updates: repeatedly
bounding only the latest step does not bound cumulative forgetting.

### 4.2 Represent missing information explicitly

Assume a retained summary satisfies

$$0\preceq G-\widetilde G\preceq\tau I.$$

This is a required invariant of a future memory component, not a property
established for the current Gaussian sketch. A two-sided spectral approximation
or a quantized basis needs its own conversion and error accounting. Numerical
rounding and representation changes must also be included.

Writing `d = w - w_ref`, define

$$D_-(w)=d^T\widetilde Gd,\qquad
  D_+(w)=d^T(\widetilde G+\tau I)d.$$

Then, simultaneously for every `w`,

$$D_-(w)\leq D(w)\leq D_+(w),\qquad
  D_+(w)-D_-(w)=\tau\|d\|_2^2.$$

This is elementary PSD order. Its simultaneous nature permits choosing a
candidate after seeing the summary, provided the matrix invariant itself is
valid for the data and any adaptive construction that produced it. A fixed-design
random-sketch theorem alone does not establish that adaptive validity.

The same estimated null direction can therefore mean either genuinely free
capacity or discarded evidence of interference. The scalar `tau` is not a task
change score; it measures what the retained state cannot resolve.

### 4.3 Acquisition has optimistic and conservative optima

Let `Q_new(w)` be a declared convex new-data objective, initially a fixed-design
multiclass squared-loss objective with an optional ridge penalty. Let `W` be a
fixed convex admissible parameter set, such as an existing basis with a trust
region. Define

$$V_{opt}=\min_{w\in W:\,D_-(w)\leq\epsilon}Q_{new}(w),$$
$$V_{true}=\min_{w\in W:\,D(w)\leq\epsilon}Q_{new}(w),$$
$$V_{safe}=\min_{w\in W:\,D_+(w)\leq\epsilon}Q_{new}(w).$$

By inclusion of the three feasible sets,

$$V_{opt}\leq V_{true}\leq V_{safe}.$$

For an acquisition target `q`:

| Evidence | Valid conclusion |
| --- | --- |
| A feasible conservative solution achieves `Q_new <= q` | An update in the existing class satisfies the declared contract. |
| A certified lower bound on `V_opt` exceeds `q` | No update in this declared class can meet the objective and contract. |
| Neither condition holds | The available evidence does not decide feasibility. |

For this affine, convex setup, both optimization problems have convex quadratic
constraints. A converged feasible solution supplies the first certificate; a
valid dual bound can supply the second. Local optimizer failure is never an
infeasibility certificate. If the parameterization uses jointly trained LoRA
factors, rank constraints, or a changing nonlinear backbone, this convex claim
does not automatically apply.

If `Q_new` is empirical loss, this certifies empirical acquisition, not held-out
generalization. Candidates need prospective evaluation on labels not used to fit
or select them, with multiplicity control for inspected candidates and trials.

### 4.4 A useful bound on the acquisition cost of compression

For one reference, suppose `W` is convex and contains `w_ref`. Let `w_opt` attain
the optimistic optimum and `rho=||w_opt-w_ref||`. For `epsilon > 0`, set

$$a=\sqrt{\frac{\epsilon}{\epsilon+\tau\rho^2}},\qquad
  w_a=w_{ref}+a(w_{opt}-w_{ref}).$$

The contracted update is conservatively feasible, since

$$D_+(w_a)=a^2D_+(w_{opt})
          \leq a^2(\epsilon+\tau\rho^2)=\epsilon.$$

Convexity gives the computable bound

$$0\leq V_{safe}-V_{opt}
 \leq(1-a)\bigl[Q_{new}(w_{ref})-V_{opt}\bigr].$$

Equivalently, the conservative learner can retain at least an `a` fraction of
the optimistic objective improvement in this model. A trust-region radius `R`
gives the uniform factor `sqrt(epsilon/(epsilon+tau R^2))`. To retain a fraction
`1-eta`, it suffices that

$$\tau R^2\leq\epsilon\bigl[(1-\eta)^{-2}-1\bigr],\quad 0<\eta<1.$$

This connects memory precision to a desired acquisition quality, rather than
selecting rank solely from explained activation variance. The derivation uses
standard convexity and is a foundation, not a priority claim. It does not handle
arbitrary multiple-reference intersections by the same contraction, nor supply
the missing nonlinear or statistical guarantees.

Increasing storage after compression cannot recover discarded historical
directions. A policy must preserve adequate information when it is observed,
use an already retained auxiliary buffer, or record that the uncertainty cannot
be reduced. No retrospective reconstruction is assumed here.

### 4.5 A concrete ambiguity example

Consider scalar linear regression in three coordinates with reference weights
zero, historical targets zero, and `0 < tau < 1`. Two histories have normalized
Gram matrices

$$G_A=\operatorname{diag}(1,\tau,0),\qquad
  G_B=\operatorname{diag}(1,0,\tau).$$

A memory interface that retains only the top eigenvector/eigenvalue, the maximum
discarded eigenvalue, the trace, sample count, and feature-label cross moments
can return the same state in both worlds:

$$\widetilde G=\operatorname{diag}(1,0,0),\quad
  \text{tail bound}=\tau,\quad\text{trace}=1+\tau,\quad X^TY=0.$$

Take a new requirement `w_2 = s`, with `s != 0`. Its minimum old-output change is
`tau s^2` in world A and zero in world B. For `0 < epsilon < tau s^2`, exact new
fit under the contract is impossible in A and feasible in B. Both worlds are
realizable with finite real-valued regression observations.

Any rule seeing only that identical state and the identical new task must make
the same decision in both worlds. If forced to issue a binary feasible/infeasible
answer, a randomized rule has worst-case error at least one half across them.
An unresolved answer is therefore sometimes necessary for this memory interface.

This is not a general bit-complexity lower bound and does not establish that all
sketching algorithms lose this particular distinction. Obtaining a useful memory
lower bound for a specified family of acquisition queries is part of the research
needed beyond this example.

## 5. What has to change for an actual IA3 learner

### Prediction drift is not identical to loss or accuracy

For exact affine predictions and squared loss, write the old reference loss as
`R_ref = E ||f_ref(X)-Y||^2 / 2`. The exact loss difference includes a residual
cross term:

$$R(w)-R_{ref}=\mathbb E\langle f_{ref}(X)-Y,\Delta f(X)\rangle
                         +\tfrac12D(w).$$

Cauchy--Schwarz gives an upper bound `sqrt(2 R_ref D(w)) + D(w)/2` when the
expectations refer to the same declared population. Dropping the cross term
without a stationarity or orthogonality argument is invalid.

If the outputs are logits with a fixed class set, reference margin at least
`m > 0`, and `D <= epsilon`, the fraction whose reference winner can change is
at most `2 epsilon / m^2`. Add the fraction of examples with margins below `m`
when a uniform margin is unavailable. This follows because a winner change
requires an output-change norm of at least `m/sqrt(2)`. It preserves the reference
decision, including any errors; it is not a blanket classification guarantee.
Introducing new competing classes changes the contract unless their logits are
included from the outset or explicitly accounted for.

### Nonlinear deployment needs a remainder budget

For updated IA3 weights, write

$$f_{w_{ref}+d}(x)-f_{w_{ref}}(x)=J_{ref}(x)d+e(x,d).$$

If a valid historical-population bound `sqrt(E ||e||^2) <= beta(d)` is available,
then actual output drift satisfies

$$\sqrt{D_{actual}(d)}\leq\sqrt{d^TGd}+\beta(d).$$

Without such a bound, the matrix result is a surrogate certificate only.
Measuring the remainder on current-task examples does not certify the absent
old population. Layerwise activation covariances also cannot replace the
end-to-end output Jacobian without an additional propagation argument.

The current target-class-probability Jacobian is not an all-logit protection
operator. Extending protection has a real derivative-computation and storage
cost. Full historical matrices are practical for a small readout but may be
prohibitive for all IA3 coordinates; runtime feasibility must be measured before
claiming an efficient algorithm.

### New bases require historical cross information

Within a fixed ambient feature map, known linear basis changes can transform
retained full moments exactly. Adding a newly learned nonlinear feature generally
requires old cross moments and targets involving that feature, which the old
summary need not contain. Preserving old adapter weights does not supply these
missing measurements.

Possible implementation choices are an explicitly fixed ambient tangent map,
an already retained feature buffer at the appropriate injection layer, or a
separate protected expert with a checked gate. Each changes the assumptions and
resource accounting. Final-backbone features cannot reconstruct arbitrary
intermediate-layer activations. Feature replay is memory and must be counted.

### Order invariance has a narrow, useful meaning

Exact sums of fixed design/target statistics are invariant to batch order, up to
floating-point effects. That does not cover changing heads, learned bases,
selected targets, adaptive assignments, or growth decisions. A streaming sketch
can obey the same error guarantee under every ordering while producing different
solutions. Preserve and test these distinctions rather than claiming that a
complete adaptive learner is order invariant.

## 6. How growth, retrieval, and learning would fit together

| Mechanism | Proposed role | Required limitation |
| --- | --- | --- |
| Learning inside an existing module | Solve an acquisition problem constrained by retained output behavior, and log optimistic/conservative objective values. | Begin in a tractable model; do not label a nonlinear local optimum globally optimal. |
| Updating memory | Preserve precision sufficient for anticipated updates, subject to the total budget. | Memory already discarded is unavailable; an unresolved state must remain an allowed outcome. |
| Increasing rank or creating an adapter | Test a larger declared class only when the existing class cannot meet the acquisition target, or as an explicitly exploratory action. | Exploration is not proof of capacity exhaustion; charge every candidate and failed probe. |
| Retrieval and OOD | Evaluate the final predictor's new gain and old risk, including its router/head. | Feature novelty does not establish benefit. A frozen old bank can still lose accuracy through new routing. |

For a separate expert, protection must cover the introduced gate and all affected
predictions. [FireGate, November 2026](https://arxiv.org/abs/2611.11979) is a direct
recent comparison for gated retention; a gate-only certificate is not the proposed
new contribution. Initially keep retrieval fixed to isolate acquisition, then
evaluate grouped and mixed sample routing as distinct protocols.

## 7. Practical study before a large server run

No experiment in this section has been started by this review.

1. **Use training boundaries for the first controlled study.** Keep test task IDs
   unavailable. This makes SD-LoRA and CL-LoRA comparable and isolates capability
   acquisition. Task-free streams are a subsequent extension with matched
   information access, not an implicit advantage for one method.
2. **Establish a numerical reference in a fixed model.** Compare exact historical
   moments, compressed moments without error control, and the proposed interval.
   Hold data, model class, targets, and budgets fixed. Check actual historical
   drift, acquisition, unresolved decisions, and dual/feasibility residuals.
3. **Test whether uncertainty is a real bottleneck.** Independently vary model
   rank and memory precision. Include a high-memory fixed-capacity control and a
   low-memory expanded-capacity control. If extra memory never changes a useful
   decision, the central mechanism lacks empirical support.
4. **Move to ImageNet-R with one isolated learning change.** Keep the existing
   stream manifest and backbone fixed for paired comparisons. Compare actual
   nonlinear output drift with the surrogate; record head and router changes.
   Full-moment readout checks are a diagnostic, not evidence of efficient
   all-layer IA3 protection.
5. **Use recent external baselines.** Start with [official SD-LoRA code](https://github.com/WuYichen-97/SD-Lora-CL)
   and [official CL-LoRA code](https://github.com/JiangpengHe/CL-LoRA), then the
   closest memory/rank controls QGPM and DYRA. Treat reimplementations or ports
   to ViT as such. Compare with BISECT when a reproducible primary implementation
   is available; do not infer a method's performance from an unverified port.
6. **Broaden only after viability.** Add recurring same-label domain shifts,
   class-order changes, and task-free boundaries. Measure new-task acquisition,
   old-task loss/accuracy, worst-cohort drift, final/average accuracy, model bytes,
   retained-statistic bytes, buffer bytes, peak memory, time, and inference cost.

The main comparator is the same learner at the same total resource budget,
with a simple growth/update policy. Counting adapter parameters while omitting
historical matrices, temporary Jacobians, reference copies, and buffers would
invalidate the claimed resource benefit. Official benchmark results and this
fork's stratified split must remain separately labeled.

## 8. What would make this a substantial contribution

The intended contribution would need all of the following, not just a new name:

- A precise acquisition-feasibility problem with incomplete historical state,
  and a meaningful lower bound showing when additional memory information is
  necessary to decide it.
- A tractable algorithm whose uncertainty region is useful at realistic memory
  budgets, with an allocation result beyond the elementary PSD/convexity bounds
  above. Explain when spending a byte on retained information is preferable to
  spending it on new parameters.
- A defensible connection to deployed adapter predictions: an exact model, a
  valid remainder/generalization argument, or an explicitly limited empirical
  claim. A surrogate proof cannot be promoted into an unconditional deep-network
  retention theorem.
- Evidence across at least two learning mechanisms, including a baseline outside
  HESTIA, that the policy improves acquisition and retention under matched total
  budgets. A changed split, improved routing alone, or fewer adapters is not enough.

Reject or narrow the direction if the bounds are routinely vacuous, the extra
state costs more than a simple competitor, ordinary updating matches the full
policy, or the apparent gain disappears when the head/router is controlled.
Also reject the priority claim if a closer paper supplies the same feasibility
decision and memory result. The present review identifies a research target; it
does not establish uniqueness, superiority, or ICML 2027 readiness.
