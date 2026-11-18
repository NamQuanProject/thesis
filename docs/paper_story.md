# When does change require new capacity?

Positioning review: 18 November 2026. Literature scope: 2025 onward.
This is a proposed paper narrative and a bounded gap analysis. It is not a
claim that the method is implemented, experimentally successful, or the first
solution to every question below.

## The decision about plug-and-play

A plug-and-play component is not a scientific requirement. If it means a
controller attached to an existing learner, its value must come from solving
a well-defined decision problem better than a simpler integrated learner. If
it means adding a new adapter, the benefit of that addition must be tested
against learning with the capacity already available.

Portability is useful evidence that the result is not a HESTIA-specific fix.
It is not the central contribution. In particular, a few observed losses from
an arbitrary black-box trainer do not reveal the best prediction that trainer's
model class could achieve. A certificate about available capacity requires
additional structure, an explicitly limited candidate set, or a tractable
optimization problem. Do not promise a universal certified wrapper.

## Proposed introduction

A continual visual learner encounters images whose appearance, category mix,
and difficulty evolve over time. Some changes alter only how familiar objects
look. Others bring back a previously encountered condition. Some expose a
limitation that the current predictor can repair through further learning,
while others require representations that the available model cannot express
without damaging earlier predictions. The learner observes examples and labels;
it is not told which kind of change has occurred. Giving every unfamiliar
window a separate adapter can preserve isolated solutions, but also duplicates
knowledge, increases selection cost, and introduces new opportunities for
routing errors. Updating one shared model indefinitely can instead sacrifice
earlier capabilities. The required action depends on the predictive consequence
of learning, not just on the novelty of the observations.

Recent work makes substantial progress on this problem. HESTIA couples
linearized adaptation with density-guided retrieval. FiUni uses Fisher geometry
to choose reuse, expansion, or new subspaces. TIDE studies expansion triggered
by measured interference, while other recent methods improve online learning
through gradient reweighting rather than a new expansion policy. These results
already establish the relevance of adaptive structure, geometry, and selective
updates. The question we pursue is more specific: when does the observed
evidence justify an expansion over the best admissible use of existing capacity?
Sources: [HESTIA](https://proceedings.mlr.press/v337/le26a.html),
[FiUni](https://arxiv.org/abs/2608.27070),
[TIDE submission](https://openreview.net/pdf?id=CA4yNpLTAU),
[Fine-Grained Hypergradients](https://arxiv.org/abs/2502.18762).

That comparison becomes difficult when past observations are unavailable.
A compressed historical basis can make an update appear harmless because the
directions on which it would interfere were discarded. Conversely, conservative
protection can block a useful update and make a model appear to need more
capacity than it actually does. Poor adaptation can therefore reflect either
insufficient capacity or insufficient information to establish what the current
capacity can safely learn. These explanations call for different actions, yet
the same retained state can be compatible with both. This is the proposed
information problem; it must be distinguished from the already studied problems
of compression-aware projection and spectral rank allocation.

The task-free setting makes this distinction operational. There is no supplied
event at which a task has been completed, an old model should be protected, or
a new module should be admitted. Regimes can overlap or recur, and a late
observation of a known class need not define a new learning problem. Thus a
mistaken structural decision changes the hypothesis class, stored information,
and inference procedure used for later decisions. An algorithm cannot rely on
true task labels to establish its retention references and then describe its
growth decision as task-free. The information available to training, memory
construction, and inference must be specified together.

We propose to study capacity expansion as a constrained comparison between
learning with existing parameters and learning with additional parameters.
Both alternatives must satisfy the same declared retention requirement and
resource accounting. The comparison should distinguish a supported update,
an expansion whose additional value is supported, and an unresolved decision
when the historical state cannot decide feasibility. A concrete technical route
is to maintain lower and upper bounds on acquisition under incomplete
retention information. These bounds would determine what the learner can
justify, rather than treating every failure of a conservative optimizer as
proof that capacity is exhausted.

The research goal is to determine when this distinction improves learning
without task boundaries: acquiring useful new prediction, avoiding unnecessary
expansion, and preserving earlier behavior after the complete predictor is
deployed. HESTIA is one test implementation; recent continual LoRA and online
task-free learners provide external controls. The required evidence is a better
acquisition--retention tradeoff at matched total cost, including historical
state and unsuccessful candidate trials. A smaller adapter count or a valid
surrogate inequality alone would not answer the question.

## What the recent literature does and does not establish

The last column states an unresolved target in this inspected comparison. It
does not assert a universal absence from all published or unpublished work.

| Candidate research gap | Closest recent overlap | Defensible remaining target |
| --- | --- | --- |
| Evidence that expansion adds capability beyond an admissible update | [FiUni, 2026 preprint](https://arxiv.org/html/2608.27070) uses subspace similarity; [TIDE, public ICLR 2026 submission](https://openreview.net/pdf/353910bd6359da726c26ac0f8533d6e34366f8ef.pdf) uses an interference-related trigger. | Bound the best attainable new-data objective in a declared existing model class under a retention constraint, and compare an expanded candidate against that bound. A new trigger by itself is not the gap. |
| Distinguishing lack of capacity from lack of historical information | [QGPM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/24e8b46430df965674221665816a4964-Abstract-Conference.html) handles compression error; [DYRA, ICML 2026 workshop](https://openreview.net/pdf/cb68cae0fe3904ee13d7cf5ca49aec01bed27141.pdf) uses spectral tails for rank allocation; [Interference and Retention, 2026 preprint](https://arxiv.org/abs/2607.09202) studies capacity, control, and memory. | Characterize which acquisition decisions cannot be identified from a given historical summary, and make the resulting uncertainty region useful for resource allocation. Geometry or compressed memory alone is not new. |
| Protecting the deployed predictor when it grows | [FireGate, 2026 preprint](https://arxiv.org/html/2611.11979) already controls behavioral change through calibrated gates, including composition across a stream. | Connect capacity admission to a protection population and reference constructed without supplied task boundaries, and include routing/head changes in the evaluated claim. Generic certified gated updates are already addressed. |
| Retention references without true task boundaries | FiUni provides boundary-free subspace decisions; [EG-CNN, Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.132665) evolves a classifier without task identifiers and includes uncertainty deferral. | Specify what acquired behavior is protected, when its reference is fixed, and what repeated adaptive decisions guarantee using only observed history. Neither task-free expansion nor deferral alone is new. |
| Stable decisions under recurrence and changing presentation order | HESTIA studies order-related adaptation; FiUni includes reuse; [BISECT, November 2026 preprint](https://arxiv.org/abs/2611.03574) addresses restrictive low-rank bases. | Separate fixed-statistic invariance from the order sensitivity of growing structure and retained memory. Test whether redundant growth persists under recurrence. Do not promise exact order invariance of an arbitrary causal learner. |

The first two rows form the recommended core. The third and fourth are required
for an honest task-free deployment claim. The fifth is an evaluation and theory
extension, not another independent module that must be added immediately.

### Claims ruled out by this review

- "No method decides when to grow." FiUni, TIDE, and other evolving architectures
  already do so.
- "No method distinguishes different types of change." This is not a defensible
  absence claim. [LargeMonitor](https://arxiv.org/abs/2606.09430) also proposed a
  detection-and-diagnosis controller. Its current arXiv record is **withdrawn**;
  it is a related proposal, not validated evidence or an accepted baseline.
- "No method considers information lost by compression." QGPM and the recent
  interference-geometry analysis are direct counterexamples.
- "A new expert cannot harm old tasks if old weights are frozen." This ignores
  the deployed router, head, and competing class outputs.
- "An unresolved state is our novelty." EG-CNN already uses deferral for
  uncertainty; the proposed unresolved state concerns architectural feasibility,
  and that narrower distinction still needs a priority audit.
- "Task-free requires adding adapters." [Fine-Grained Hypergradients, 2025
  preprint and TMLR 2026 publication](https://openreview.net/pdf?id=Tu9rHiczLm)
  studies online task-free adaptation through prototypes and gradient balancing.
  Expansion must beat strong update controls; it is not required by the problem
  definition.

## The precise contribution to aim for

The story needs an explicit comparison, not just an intuition that a new module
looks useful. At time `t`, let `F_keep` be a declared family of deployed predictors
achievable using existing capacity and the permitted update protocol. Let
`F_expand` include an additional parameter allocation. Both have the same
available training information. Write

$$V_t^{keep}=\inf_{f\in\mathcal F_{keep}:\,D_{old}(f)\leq\epsilon_t}R_t(f),$$
$$V_t^{expand}=\inf_{f\in\mathcal F_{expand}:\,D_{old}(f)\leq\epsilon_t}R_t(f).$$

`R_t` is risk on a specified current distribution or evaluation window;
`D_old` is a specified old-risk or output-drift contract with a declared
reference. They are not interchangeable. Costs and fitting opportunities must
be specified separately; these oracle values do not reveal an untrained
candidate's performance to the algorithm.

The comparison of interest is `V_keep - V_expand`, with resource costs included.
Improving over the currently frozen incumbent only measures one point inside
`F_keep`, and can overstate the need for additional parameters.

For a concrete expanded predictor `f_plus`, a sufficient admission target is

$$L_t^{keep}-U_t^R(f_{plus})>\gamma+\lambda\Delta B_t,
  \qquad U_t^D(f_{plus})\leq\epsilon_t,$$

where `L_keep` is a valid lower bound on `V_keep`, `U_R` bounds the candidate's
new risk, `U_D` bounds its protected drift or excess risk, and `Delta B` is the
declared resource increment. All quantities must refer to compatible objectives,
information, and units. This is a target guarantee, not an implemented or proved
general algorithm. Future usefulness cannot be certified under unrestricted
future drift.

The resource charge uses explicitly declared loss-equivalent prices; hard memory
and compute limits are enforced separately. Candidate construction and failed
trials remain costs of the full policy. Give the ordinary-update comparator the
same permitted fitting expenditure rather than an artificially weak budget.

The [acquisition and memory note](acquisition_research_2025plus.md) gives a
tractable starting point with affine predictors: incomplete historical moments
lead to nested feasible sets and optimistic/conservative optima. Its basic
PSD-order and convexity arguments are standard. A substantial new result would
need a nontrivial information or allocation theorem, useful bounds, and an
algorithm that realizes them under the declared task-free information protocol.
If only a finite set of trained alternatives can be compared, state that scope
instead of claiming to certify exhaustion of an entire nonlinear model class.

Three proposed contribution statements for a future completed paper are:

1. **A capacity-admission problem without task-boundary supervision.** Formalize
   additional predictive value relative to learning within existing capacity,
   under explicit retention and total-resource constraints. Distinguish it from
   detecting a distribution change or merely observing forgetting.
2. **An information limit and a computable decision rule.** Characterize when
   incomplete historical state makes update feasibility undecidable, derive
   informative bounds in a tractable regime, and allow uncertainty instead of
   misclassifying it as exhausted capacity. The current small ambiguity example
   is a starting point, not yet the required general result.
3. **Evidence that the decision improves continual learning.** Demonstrate better
   new-capability acquisition and old-behavior retention at matched total cost,
   across HESTIA and at least one recent external learner. Include the complete
   inference path and all probe/memory costs.

These are deliverables to establish. The present repository does not justify
writing "we prove" or "we outperform" for this complete contribution package.

## Why task-free is essential to this version of the story

Task boundaries provide organizational information, although they do not by
themselves prove that another module is useful. A benchmark can otherwise tell
the learner when to snapshot a reference, assign examples to a memory summary,
reset an optimizer, or allocate a module. Removing boundaries makes those events
decisions of the learner. Our method must not restore the same oracle through
the retention mechanism or candidate scheduler.

For the proposed primary setting, assume supervised labels become available
after predictions; do not imply unsupervised discovery. Define a causal rule for
reference creation and historical sampling before running the evaluation. A
pooled history mixture is a possible protection target, but can hide rare-regime
damage. Protecting every absent regime requires extra assumptions or retained
coverage, and must not be claimed from a pooled bound.

If the eventual algorithm still needs true boundaries to maintain its certified
state, the correct paper is about constrained continual adaptation, with a
task-free extension left open. Conversely, a strong task-free result must show
that its structural decisions work when boundaries are removed, blurred, and
revisited, not just report a score on disjoint class blocks.

The earlier note's recommendation to start with known training boundaries remains
a useful implementation diagnostic. It would not, by itself, establish the
task-free paper proposed here.

## Evidence that would make the narrative convincing

Use a controlled diagnostic with an exact constrained oracle before a deep-model
benchmark. For that oracle, an unnecessary expansion is one made when the
existing declared class could already achieve the target under the contract;
a missed opportunity is a feasible useful action not taken. On nonlinear
ImageNet-R models, a menu of fitted alternatives is only an empirical comparator,
not proof of the full class optimum.

ImageNet-R remains the principal image benchmark. In addition to its standard
class stream, construct and disclose controlled repeated-class and gradual/mixed
streams. Do not assume it supplies ground-truth style labels for free. Include
these mechanisms separately:

| Controlled condition | What a useful policy should demonstrate |
| --- | --- |
| Changed appearance with little predictive damage | Avoid paying for growth without measurable benefit. |
| Returning conditions already handled by stored knowledge | Reuse or repair selection without automatically creating duplicates. |
| New predictable structure that the current constrained class cannot fit | Admit additional capacity when a feasible candidate supplies the missing gain. |
| Noise or ambiguity without an informative predictive cue | Avoid claiming that more parameters necessarily solve the error. |
| Same capacity with increasingly compressed historical memory | Distinguish uncertain retention from genuinely impossible constrained learning. |

The first two conditions are hypotheses about the actual predictors, not facts
guaranteed by the names of the shifts. Candidate fitting, prospective evaluation,
and assessment must use separate permitted information.

Compare the full decision rule with (a) an equally funded ordinary update,
(b) novelty-triggered growth, (c) interference-triggered growth, and (d) the same
retention mechanism without the capacity-admission test. Recent external controls
include SD-LoRA and CL-LoRA under their supported protocols, Fine-Grained
Hypergradients for online task-free adaptation, and FiUni or a clearly labeled
port for geometric decisions. A task-aware baseline must not silently receive
more boundary information in a task-free comparison.

Measure acquisition, retention, unnecessary expansion in the oracle diagnostic,
unresolved decisions, eventual recovery on recurrence, and all persistent and
temporary memory. Fix and disclose training/test batch composition. A batch
drawn from one hidden regime supplies different inference information from a
mixed batch. Do not pool their results.

If a simple update performs equally well at the same cost, there is no evidence
that an additional controller is needed. If FireGate-style protection plus an
existing expansion trigger matches the complete system, the proposed decision
must offer a further demonstrable advantage or the novelty claim should narrow.

## Review and artifact status

This update added a narrative, inspected recent primary sources, and recorded
additional close work. TIDE was inspected through indexed primary PDF passages;
direct OpenReview access was blocked, and its acceptance status is not asserted.
Its indexed method figure describes task-assigned adapters and a memory buffer;
it is relevant growth prior work, not a verified boundary-free comparator under
our proposed information protocol.
The LEViT and EG-CNN publisher records were inspected: they further rule out a
first task-free-expansion claim. LargeMonitor's withdrawn status was checked on
arXiv. Recent source descriptions are not independent reproductions of results.

No new model method, data stream, benchmark result, remote experiment, or
independent reviewer endorsement was produced in this update. The README remains
limited to running the code. A working title for the proposed question is
**When Does Change Require New Capacity?**; a title is not evidence of novelty.
