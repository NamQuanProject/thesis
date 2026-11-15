# Research direction: specialization without hard task exclusion

Working title: **Specialize Without Excluding: Bounded Probability Corrections
for Task-Free Continual Learning**. Updated 2026-11-15. This is an implemented
research candidate, not a claim of priority, benchmark superiority or ICML readiness.

## The scientific question

Can a growing collection of local predictors improve a global classifier while
bounding the damage from incompatible specialists and avoiding an inference pass
through every expert? The intended setting permits overlapping observed label
scopes and arbitrary mixtures of sources within a test batch. It supplies neither
test task IDs nor old images for replay.

In this checkout, hard HESTIA routing selects a segment and sets every other class
logit to minus infinity. A routing mistake can therefore give the true class zero
probability. A batch of one ground-truth task supplies useful collective evidence;
that advantage need not survive arbitrary batching. Also, frozen adapters currently
share a changing classifier head. These are concrete implementation observations,
not discoveries of previously unknown continual-learning failure modes.

The coupled tangent solver is a separate adaptation ablation. This direction keeps
the original block solver to isolate inference. It stores each completed adapter
with its classifier head, builds a global classifier from additive statistics of
clean pretrained features, and replaces hard selection with support-local updates.
The existing convergence/new-label detector still determines segment creation.
Consequently this implementation does NOT solve general unlabeled concept drift.

## Primary-literature audit and overlap

| Primary source | Existing contribution we must credit | Distinction being investigated here |
|---|---|---|
| [HESTIA, UAI 2026](https://proceedings.mlr.press/v337/le26a.html) | Linearized adaptation and density-guided adapter retrieval. | Replace hard exclusion at inference; keep its training as a controlled candidate generator. |
| [Kim et al., NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/20f44da80080d76bbc35bca0027f14e6-Paper-Conference.pdf), [ICML 2023](https://proceedings.mlr.press/v202/kim23x.html) | Within-task and task-prediction decomposition, OOD connection and learnability. | No claim to invent the routing problem or conditional probability factorization. Detected scopes may overlap. |
| [MOS, AAAI 2025](https://arxiv.org/html/2412.09441) | Adapter merging, self-refined retrieval and multi-stage ensembling. | Explicit probability floor and full-model decision certificate, rather than a self-consistency retrieval heuristic. |
| [FDC, September 2026 preprint](https://arxiv.org/html/2609.34503) | Post-hoc task-head calibration using subspaces, residual likelihood and class affinity. | A constrained probability operator for overlapping scopes, not another density/affinity score. FDC is a necessary later empirical comparison. |
| [Hinton et al., 2015, Section 5](https://arxiv.org/pdf/1503.02531) | Generalist plus overlapping specialists, selection and KL reconciliation with a dustbin. | No claim that generalists, specialists, overlap, or soft combination are new. Current candidates need no outside-scope training examples or dustbin; this makes their out-of-scope outputs untrusted. |
| [Li and Lin, IJCAI 2017](https://arxiv.org/pdf/1709.02123) | Integrate specialized classifiers through class transitions and a continuous-time Markov chain equilibrium. | A bounded single correction to a retained global prediction, with classwise budgets and partial-evaluation guarantees. Class-mass transport itself is NOT a first-use claim. |
| [Wang et al., QWYC](https://arxiv.org/pdf/1806.11202) | Optimize ensemble evaluation ordering and early stopping. | Exploit known label supports and mass budgets for a deterministic certificate; early-exit ensembles and remaining-contribution bounds are established ideas. |

The review inspected primary abstracts and relevant full-text sections for the
closest specialist, calibration, decomposition and early-exit papers. It is not
an exhaustive priority review. The potential contribution is the **joint treatment
of classwise interference budgets, overlapping continually acquired specialists,
and selective inference**. Its value must survive the ablations below. A different
name or an elementary proof does not establish a sufficiently original paper.

## Operator and implementable choices

For one input, let p0 be a strictly positive global probability vector on C classes.
Expert e has an observed class set S_e and a probability vector q_e on S_e. Let
m_e = sum_{y in S_e} p0_y. Select nonnegative weights eta_e and define

    rho_y = sum_{e: y in S_e} eta_e <= alpha < 1,
    p_y = (1-rho_y) p0_y + sum_{e: y in S_e} eta_e m_e q_e(y).

Every expert redistributes probability only inside its scope. For any selected
subset E, use the same formula with sums restricted to E. Unevaluated experts
then act as identity, q_e(y)=p0_y/m_e. Importantly the implementation does not
recompute the weights after selecting a subset.

The initial implementation uses eta_e = alpha / max_{y in S_e} degree(y), where
degree counts nontrivial experts containing y. A singleton implements identity
and is removed from computation. This is a simple feasible allocation, not an
optimal allocation theorem. Disjoint scopes can each use alpha; there is no
global 1/number-of-experts dilution. Overlap appropriately consumes a shared
budget. Other feasible input-dependent weights could be studied later; they
must be fixed before evaluating the unknown expert outputs for the certificate.

The global classifier is established streaming LDA: additive per-class counts,
sums and one shared second moment, OAS covariance, uniform class prior. It uses
the existing Gaussian code with a fixed 1e-4 covariance diagonal floor. Features
come from the frozen pretrained encoder and the deterministic test transform of
TRAINING images. No validation/test observation enters these statistics. A
1e-8 uniform probability floor is part of the predictor, not silent loss clipping.

Initial alpha=0.8, anchor temperature=1, budget cap=2 are predeclared feasibility
settings, not test-selected optima. The alpha choice allows a worst-case excess
log loss of log(5), which is a finite but loose bound. Increasing alpha trades a
weaker guarantee for more influence by specialists. Head snapshots cost O(ECD)
in the initial implementation; the global statistics cost O(D^2+CD). Raw images
are not stored. The implementation has a growing expert bank, not bounded total
memory. No claim of new Gaussian estimation or order-invariant full training is made.

## Mathematical development

These are deterministic derivations for the stated operator. They use elementary
probability and convexity. Their application is the research claim; they are not
presented as new general concentration, ensemble or Markov-chain theory.

### 1. A sharp classwise robustness condition

Fix p0 with positive entries, scopes of size at least two, weights eta>=0 and
alpha in [0,1). For every possible collection of expert distributions, the
operator is normalized and satisfies p_y >= (1-alpha)p0_y for all y **if and
only if** rho_y<=alpha for all y.

Proof of sufficiency: summing over y removes sum_e eta_e m_e and adds exactly
the same quantity because each q_e sums to one. Each contribution is nonnegative,
so p_y >= (1-rho_y)p0_y >= (1-alpha)p0_y. For necessity, if rho_y>alpha, choose
every expert covering y to put all of its mass on another class in its scope.
Then p_y=(1-rho_y)p0_y < (1-alpha)p0_y. The exclusion of singleton experts is
essential for this last construction; singleton corrections are identically zero.

It follows pointwise, for any true class y, that

    -log p_y - (-log p0_y) <= -log(1-alpha).

The result holds for ANY selected subset with the same feasible weights. A
specialist whose scope excludes y contributes exactly zero to the change of p_y.
That is not a guarantee of top-1 correctness: a competing class can gain mass.
Nor is it a lifelong forgetting bound. Both p0 and feasible weights can change
as the bank grows, so locality for a fixed bank does not mean all historical
probabilities remain fixed after further training.

### 2. Exact remaining-work certificate, including overlapping scopes

Fix a bank, p0, and its weights. Let E be the experts already evaluated and set

    L_y = (1-rho_y)p0_y + sum_{e in E: y in S_e} eta_e m_e q_e(y),
    U_y = L_y + sum_{e not in E: y in S_e} eta_e m_e.

For the fully evaluated predictor p*, L_y<=p*_y<=U_y. Therefore, if

    L_c > max_{j != c} U_j,

c is the unique full-model top-1 decision, regardless of all unevaluated outputs.
The anchor completion used to return a normalized partial distribution is one
such completion, so its top-1 is also c. Its probabilities, NLL and top-5 need
not equal those of the full model. The code uses float64 bounds and a conservative
roundoff slack; full exhaustion handles ties by the usual deterministic argmax.

This certificate is sharp for the information model in which each unevaluated
expert can be any point of its known simplex. For any rival j, the minimum
possible difference p*_c-p*_j is L_c-U_j: put each unknown expert covering j
entirely on j, and each other expert covering c on a different supported class.
All choices are independent and feasible. Thus if the strict inequality fails,
one cannot certify c as the unique winner from just the observed outputs,
supports and weights. This is NOT a universal lower bound on neural inference;
additional bounds on specific neural outputs could certify earlier.

### 3. Why support matters for computation

An expert cannot increase the score of a class outside its support. For disjoint
scopes the unknown increase of a rival is bounded by its scope's global mass,
instead of the sum of every unevaluated expert's weight. This creates a testable
mechanism for early stopping even when the bank contains many irrelevant experts.
The implementation queries the largest eta_e m_e per unresolved image and groups
identical queries for GPU execution. This ordering is a heuristic; certificate
validity does not require it to be optimal. A cap of two may stop without a
certificate and must report its uncertified fraction and disagreement with full.

Batch composition cannot enter the mathematical prediction because all quantities
are functions of one input and a fixed bank. GPU roundoff can affect near ties;
the ablation script saves per-example traces and explicitly counts disagreements.

## Falsifiable ImageNet-R study

Train one bank with the original HESTIA block update, the same stream, seed,
backbone and split as controls. Freeze each segment's head at its own completion.
Evaluate these predeclared modes on the SAME checkpoint and examples:

| Mode | Question |
|---|---|
| Global anchor only | Are any gains just the established global Gaussian classifier? |
| Scope-mass-weighted soft mixture + anchor | Is a conventional convex mixture sufficient? |
| Full support-local operator | Does constrained specialization add predictive value? |
| Certified evaluation | How much expert work can be skipped with zero certified top-1 disagreement? |
| At most two experts per image | What is the explicit accuracy/work tradeoff when certification is unavailable? |

Each runs with ordinary grouped and randomly mixed validation batches, without
fitting on either. Record macro accuracy, per-source accuracy, NLL, normalized
Brier loss, image-expert evaluations, certificate frequency and actual wall time.
Per-example traces verify the identity of the evaluation populations, lazy/full
agreement and batching invariance. The script nulls forgetting for post-hoc modes
because their own historical accuracy matrices were not measured. Full training
records a valid forgetting matrix for its declared inference mode.

HESTIA batch/sample and class-Gaussian batch/sample are separate controls. A new
training run is needed because old checkpoints lack the per-segment heads and
global clean-feature statistics; reconstructing them from the final head would
misrepresent the method. The anchor-only inference ablation shares a bank-building
run's training cost. A future standalone anchor training control is required for
training-cost comparisons. FDC, MOS, CTMC aggregation and stronger calibrated
mixtures remain necessary comparisons before making a broad empirical claim.

Reject the proposed performance story if the full operator fails to improve over
the anchor/ordinary mixture, if improvement comes only from giving it extra data,
or if useful accuracy requires evaluating almost every expert. A probability-floor
theorem is not evidence that accuracy improves. Initial one-seed runs assess
viability; they are not a significance study. Freeze choices before paired seeds
15/30/45 and do not tune from these final test reports. If the method survives,
extend to recurring/overlapping training streams and another adapter learner;
overlap unit tests alone do not establish those empirical claims.

## Run on the prepared server

Local validation on 2026-11-15: 41 tests passed across mathematical invariants,
legacy baselines, coupled controls, tiny custom-ViT training/checkpoint roundtrips,
and the ten-mode paired inference script. Those checks establish mechanics only;
there is no completed ImageNet-R result for this candidate at this point.

```bash
python scripts/run_imagenet_r.py --stage pilot --variants mass --seeds 15 \
  --data-path /workspace/ContinualLearningThesis/data \
  --output-root /workspace/ContinualLearningThesis/results/mass-study \
  --jacobian-mode per_sample --jacobian-chunk-size 0 --efficient-runtime --execute

python scripts/run_imagenet_r.py --stage compare --variants mass --seeds 15 \
  --data-path /workspace/ContinualLearningThesis/data \
  --output-root /workspace/ContinualLearningThesis/results/mass-study \
  --jacobian-mode per_sample --jacobian-chunk-size 0 --efficient-runtime --execute

python scripts/evaluate_specialists.py \
  --run-dir /workspace/ContinualLearningThesis/results/mass-study/compare/mass/seed15 \
  --output-dir /workspace/ContinualLearningThesis/results/mass-study/paired-evaluation
```

Use the shared venv, immutable commit checkout and supervisor wrapper described
in the server runbook. Already running source checkouts must not be edited. Pilot
scores are diagnostics, not benchmark results. Concurrent-server wall times are
exploratory; report isolated timing before making speed claims.
