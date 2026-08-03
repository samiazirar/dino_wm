# DinocularWorldModel — human plan

## Research question

Does a learned 3D-aware visual representation, such as DINOcular, help a robot
world model predict what happens after an action and plan actions that complete
the task?

The study also asks a narrower question: within DINOcular, does supplying useful
estimated depth at use time help beyond using the same learned representation
with a depth input that contains only zeros?

## The comparison

Three systems are compared:

- **DINOv2:** a color-image representation with no depth input.
- **DINOcular:** a learned 3D-aware representation supplied with color and
  estimated depth.
- **Zero-depth DINOcular:** the same frozen DINOcular encoder architecture and
  weights, but with every supplied depth value replaced by zero.

These systems answer three related questions:

- DINOcular versus DINOv2 tests whether the complete learned 3D-aware system is
  more useful than the color-image baseline.
- DINOcular versus zero-depth DINOcular tests whether informative depth at use
  time adds value within the DINOcular setup.
- Zero-depth DINOcular versus DINOv2 shows whether the learned DINOcular setup
  remains useful when its current depth input carries no information.

Zero-depth DINOcular is not an ordinary color-only model. It may also receive an
input unlike those used when its encoder was learned. The comparison therefore
does not isolate every architectural difference between DINOcular and DINOv2.

The four tasks are PushT, Wall, Rope, and Granular. **PushT is primary because
it was chosen before results, is the established DINO-WM planning benchmark,
has the largest dataset, and has a fixed 50-target success evaluation.** This
prevents choosing whichever task later looks best. PushT is not the task with
the strongest real depth variation. Wall is a flatter-scene control. Historical
real-depth DINOcular results for Rope and Granular are excluded, and their
zero-depth histories remain pending exact functional reuse proof. Their
replacement per-frame depth was regenerated on 31 July 2026 and now passes the
fixed quality check at full size. However, the direct corrected-versus-old
comparison failed its fixed moving-versus-static separation requirement, so
the corrected Rope and Granular depth remains unusable. Rope and Granular
remain exploratory because each has only 10 planning targets and no currently
accepted common binary success definition.

## The 36 training runs

One run means training one system on one task with one random seed:

| Task | DINOv2 | DINOcular | Zero-depth DINOcular | Total |
|---|---|---|---|---:|
| **PushT** | seeds 1, 2, 3 | seeds 1, 2, 3 | seeds 1, 2, 3 | 9 |
| **Wall** | seeds 1, 2, 3 | seeds 1, 2, 3 | seeds 1, 2, 3 | 9 |
| **Rope** | seeds 1, 2, 3 | seeds 1, 2, 3 | seeds 1, 2, 3 | 9 |
| **Granular** | seeds 1, 2, 3 | seeds 1, 2, 3 | seeds 1, 2, 3 | 9 |
| **Total** | 12 | 12 | 12 | **36** |

This gives 12 matched triplets, one per task and seed. The three runs in each
triplet use the same split, data order, downstream initialization, and
non-system settings. Comparisons are made within triplets before summarizing
across seeds.

All runs use total batch size 32. Training lengths are 123,858 updates for
PushT, 143,910 for Wall, and 53,500 each for Rope and Granular. Later prediction
and planning evaluations do not add training runs. PushT and Wall use 50 fixed
planning targets per model; Rope and Granular use 10, producing 1,080
model-target planning results.

## What data are used

Training is offline: the robot does not gather new experience while the models
learn. Each dataset is a collection of already-recorded trajectories. A
trajectory contains an ordered sequence of color images, the actions taken,
the robot or object state, and episode information. The DINOcular systems also
look up one depth map for the same episode and frame.

- **PushT:** 18,685 training trajectories with 2,336,736 frames, plus 21 fixed
  validation trajectories with 2,514 frames.
- **Wall:** 1,920 trajectories of 50 action-aligned frames, giving 96,000
  frames.
- **Rope:** 1,000 trajectories of 20 frames, giving 20,000 frames.
- **Granular:** 1,000 trajectories of 20 frames, giving 20,000 frames.

The released splits and episode order are fixed before training. Color and
depth are aligned and cropped to the same 224-by-224 view. DINOv2 ignores
depth; DINOcular receives the stored values; zero-depth DINOcular checks the
same selected depth record and then replaces its values with zeros immediately
before the encoder uses it.

PushT and Wall are visually flat tasks, so they test whether the complete
representation helps even when scene depth is weak. Rope and Granular keep their
exploratory perspective-depth role. Their replacement depth passes its general
quality checks but failed the fixed moving-versus-static separation requirement
and therefore cannot yet be used.

## How one run is trained

The pretrained visual encoder is frozen: the study does not retrain DINOv2 or
DINOcular. It trains a new world-model predictor and new action/state modules
for each system, task, and seed. In plain terms:

1. Take a short ordered window from a recorded trajectory. PushT uses three
   previous observations; the other tasks use one.
2. Convert each image—and depth where applicable—into visual features with the
   frozen encoder.
3. Give the predictor those features together with the recorded action and
   robot/state information.
4. Train it to predict the visual features that should come next.
5. Repeat with fixed data order until the task's exact update count is reached.

PushT and Wall advance five recorded frames per model step; Rope and Granular
advance one. The predictor learning rate is fixed at 0.00005, and image
reconstruction is disabled so the model is trained and judged in feature
space.
Within every matched triplet, the data, ordering, fresh-module initialization,
training length, and all other settings are the same; only the visual system
changes.

Each run saves the predictor, optimizer, random state, data position, and
current step so an interruption can resume exactly. After training, the model
first predicts held-out trajectory segments and then plans actions toward fixed
task targets. Evaluation targets never tune the model or alter the protocol.

## How the results will answer the question

The paper has two decision-bearing comparisons.

**Complete-system question.** PushT planning success is primary for DINOcular
versus DINOv2 because their prediction spaces differ. DINOcular meets the
predeclared utility criterion only if:

- its PushT planning-success estimate is at least 0.10 higher;
- the paired 95% uncertainty interval has a lower end above zero;
- the planning difference is positive in all three seeds;
- pooled normalized rollout error is lower at horizons 5 and 10;
- Wall rules out a planning decrease larger than 0.05; and
- PushT and Wall pass the same convergence check.

Normalized rollout error compares prediction error with copying the latest
observation forward. It is supporting information across encoders, not the
primary cross-encoder measure.

**Informative-depth question.** For PushT and Wall separately, DINOcular must
beat zero-depth DINOcular in overall planning success and in every paired seed,
with the paired 95% interval above zero. Horizon-1 normalized rollout error and
terminal state error support this comparison.

Rope and Granular are reported separately. Their task-specific errors are not
pooled with PushT or Wall. A cross-task depth claim is allowed only if valid
binary success definitions for both deformable tasks were fixed before model
evaluation.

Uncertainty uses 10,000 paired resamples of seeds and fixed evaluation targets.
Three seeds test consistency in these trained models, not every possible future
seed. All tasks, measures, nulls, and negative results remain visible. A failed
run may resume or repeat only the same task, system, seed, and settings.

All 36 checked runs are required for the complete-matrix paper claim.

Every PushT result involving DINOcular must state that its fixed recovered
depth proxy is not physical distance and does not reconstruct the original
depth-producing system.

## What we know now

All 36 planning specifications are fixed. Together they define 1,080 expected
model-target planning results. Rope and Granular use exactly 20 high-level
actions: four replanning rounds of five actions each, with no early stop based
on the outcome. Their terminal Chamfer distance is measured after action 20.

Wall DINOcular seed one and Wall zero-depth DINOcular seed one are fully
trained at the exact target of 143,910 steps. Both are ready for the fixed
held-out prediction and planning evaluations.

PushT DINOcular seed one and PushT zero-depth DINOcular seed one are healthy at
about 64% of training. Their latest measured steps are 78,724 and 78,927,
respectively, and each has exactly one unique continuation so that the same
lineage is completed rather than duplicated.

Rope DINOv2 seed one and Granular DINOv2 seed one remain fully trained at
53,500 steps. Each has completed 100 held-out prediction episodes.

The direct corrected-versus-old depth comparison has now run for Rope and
Granular. The corrected values changed and improved, but the fixed admission
requirement still failed on moving-versus-static separation: 1.402 for Rope
and 1.646 for Granular. Corrected Rope and Granular depth therefore remains
unusable until this low separation is causally resolved or its source-video
cause is measured directly.

There are 997 repeated-depth occurrences in Rope and 999 in Granular. Every
one corresponds to exactly repeated aligned color images. This shows that the
depth estimator is not repeating while the visible image changes; the
duplication comes from duplicated source-video frames. It does not yet explain
the separate low moving-versus-static separation result, because RGB motion
has not been measured directly for that issue.

Historical real-depth DINOcular results for Rope and Granular remain excluded.
Their zero-depth histories remain unusable unless exact functional reuse is
shown. No matched scientific comparison, aggregate decision, or winner exists
yet.

The empirical paper retains the current framing, methods, limitations, and
strict result-ingestion path. Missing or incomplete results produce no paper
conclusion. Rope and Granular depth results cannot support claims while their
corrected depth remains unusable, and every PushT result involving DINOcular
must retain the recovered-depth-proxy qualification stated above.

## What we know and use

| Item | Why it matters | What we know | Next useful result |
|---|---|---|---|
| **Wall seed one** | Gives the flatter-scene control for both the complete-system and informative-depth questions. | DINOcular and zero-depth DINOcular are fully trained at 143,910 steps. | Run their fixed prediction and planning evaluations. |
| **PushT seed one** | Supplies the primary planning comparison and the main informative-depth result. | DINOcular is at 78,724 steps and zero-depth DINOcular is at 78,927 steps; both are healthy at about 64%, with one unique continuation each. | Complete both lineages, then run their fixed prediction and planning evaluations. |
| **Rope and Granular DINOv2 seed one** | Supplies the color-image baseline for the exploratory deformable tasks. | Both are fully trained, and each has 100 held-out prediction episodes completed. | Preserve these results for later matched comparisons. |
| **Rope and Granular corrected depth** | Determines whether the exploratory DINOcular comparisons can use informative depth. | Corrected depth improved over old depth but failed moving-versus-static separation at 1.402 for Rope and 1.646 for Granular. All 997 Rope and 999 Granular repeated-depth occurrences align with exactly repeated RGB source frames, but RGB motion has not yet been measured for the low-separation issue. | Directly analyze RGB motion to measure the source-video cause of the remaining separation failure. |
| **Empirical paper** | Will report the fixed design and only the conclusions supported by the complete results. | No matched scientific comparison, aggregate decision, winner, or conclusion exists yet. | Generate supported tables, figures, and conclusions only after the required fixed evaluations exist. |

## Current milestone

The immediate priority remains one complete matched seed across all four tasks,
with DINOv2 versus DINOcular as the main comparison. The numeric seed label has
no special scientific status. Wall now has two newly completed DINOcular
lineages ready for evaluation, while the two PushT DINOcular lineages are the
next training completions. Rope and Granular corrected depth cannot be used
unless the low moving-versus-static separation is causally resolved or its
source-video cause is measured directly.

The three next useful actions are:

1. Run the fixed Wall prediction and planning evaluations.
2. Complete and evaluate the two PushT seed-one lineages.
3. Directly analyze RGB motion for the remaining Rope and Granular separation
   issue.

The later milestone is the complete 36-lineage matrix: all three systems on all
four tasks with all three seeds, followed by the predeclared comparisons,
figures, and empirical paper results. Training progress alone does not answer
the research question.

## Current decisions

- All 36 lineages remain limited to one computing job in each task-and-system
  chain and at most 12 project GPUs in use.
- All nine runs for one task stay on one GPU model; resumed runs retain that
  model.
- Each completed lineage proceeds to the same fixed held-out prediction and
  planning evaluations.
- The corrected Rope and Granular depth is not usable merely because it
  improved over the old depth or because exact-RGB duplication has been found.
  The separate low moving-versus-static separation must be causally resolved or
  its source-video cause measured directly first.
- Seeds two and three remain part of the fixed design; all 36 checked runs are
  required for the complete-matrix paper claim.
- No scientific claim is made before the fixed evaluations and predeclared
  comparisons support it.

## Work meter

Direct effort is the ongoing 36-lineage training campaign and its fixed
prediction and planning evaluations. Support work remains bounded. No exact
productive-work percentage is claimed without measured accounting.
