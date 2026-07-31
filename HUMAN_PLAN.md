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
zero-depth histories remain pending exact functional reuse proof. Corrected
canaries for replacement per-frame MapAnything depth passed, and full
replacement depth production finished on the evening of 30 July 2026. PushT,
Wall, and DINOv2 continue
unchanged. Rope and Granular remain exploratory because each has only 10
planning targets and no currently accepted common binary success definition.

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
representation helps even when scene depth is weak. Rope and Granular retain
their exploratory perspective-depth role only after their corrected DINOcular
full replacement depth caches pass frozen validation.

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

## Current evidence

All 36 training lineages have been submitted. Their controller allows at most
one computing job at a time in each task-and-system chain and at most 12
project GPUs overall. A lineage that completes training automatically proceeds
to its fixed held-out prediction evaluation and then to fixed planning.

All 36 planning specifications and launch wrappers are ready. Together they
define 1,080 expected model-target planning outcomes. Rope and Granular use
exactly 20 high-level actions: four replanning rounds of five actions each,
with no early stop based on the outcome. Their terminal Chamfer distance is
measured after action 20.

Granular DINOv2 seed one is the first controller-accepted fully trained
lineage. It reached 53,500 steps, and its fixed held-out prediction evaluation
completed with 100 episode records.

Rope DINOv2 seed one has now also been accepted at its exact 53,500-step target
without retraining. Its fixed held-out prediction evaluation is complete with
100 episode records.

Historical real-depth DINOcular results for Rope and Granular are excluded.
Their zero-depth histories are not accepted unless exact functional reuse is
shown. Corrected canaries for replacement per-frame MapAnything depth passed.
About 7.36 GB of obsolete real-depth-derived artifacts were permanently removed
without touching zero-depth, DINOv2, PushT, Wall, raw sources, or healthy runs.
Full replacement depth production ran in four isolated literal-singleton
MapAnything shards per task and finished on the evening of 30 July 2026. All
four shards for each task are complete, together covering the full released
data: 1,000 trajectories and 20,000 frames for Rope and the same for Granular.
The merge and validation steps first prepared for those shards were left
waiting on an earlier failed production attempt and could never have run; they
were cancelled on 31 July and resubmitted against the shards that actually
completed. The merged caches must still pass frozen full-cache validation
before their admission, any zero-depth reuse decision, or corrected real-depth
training from step zero. Neither task has an admissible DINOcular planning
outcome.

The confirmed Weights & Biases dashboard is
<https://wandb.ai/rlp_uni_bonn/dinocular-wm-campaign>. The lineage currently
visible there was imported from existing records. Imported history and live
telemetry are kept distinct, so an imported run is not presented as live
training telemetry.

PushT, Wall, and DINOv2 lineages continue unchanged. No duplicate execution or
later-seed advance is reported.

No accepted matched DINOv2-versus-DINOcular task pair is complete, and
therefore no scientific comparison or winner can yet be reported.

The empirical paper now contains the current framing, Methods, a vector
overview figure, Limitations, and a fail-closed path that will ingest the real
prediction and planning tables and figures only from the complete 36-lineage
evaluation bundle. It also applies the predeclared comparison and convergence
rules to generate machine-readable and manuscript-ready decision summaries,
including null or negative outcomes. The active controller now supplies the
fixed convergence record and will invoke this complete path after all
evaluations exist. Missing or incomplete results create no paper result
artifacts. The paper shows partial controller-accepted seed-one Results in
Progress values, including valid DINOv2 prediction and planning rows, but no
aggregate decision, winner, or conclusion exists; excluded or provisional Rope
and Granular depth rows cannot support claims. It is not published. The
obsolete PaperPilot diff is retired.

## Data and evidence map

**Observed now** means a checked fact; **next observable action** means the
event that would change it.

| Item | Why it matters | Observed now | Next observable action |
|---|---|---|---|
| **Fixed datasets and depth inputs** | Supply the color, action, state, and depth inputs for the four tasks. | PushT, Wall, and DINOv2 inputs remain fixed. Historical Rope and Granular real-depth DINOcular inputs are excluded, zero-depth reuse remains unproven, corrected canaries passed, and full replacement depth production finished on 30 July 2026 with all shards complete. Merge and frozen full-cache validation are now running. | Pass frozen full-cache validation, then admit the caches before a zero-depth reuse decision or corrected real-depth training from step zero. |
| **Training campaign** | Produces the 36 trained world models needed for matched comparisons. | Granular and Rope DINOv2 seed one are fully accepted; PushT, Wall, and DINOv2 continue unchanged. | Complete the first accepted DINOv2–DINOcular task pair. |
| **Held-out prediction** | Measures how well each trained model predicts unseen trajectory segments. | Granular and Rope DINOv2 seed one each have a complete 100-episode prediction evaluation. | Evaluate the next completed lineage under the same fixed procedure. |
| **Fixed planning** | Tests whether prediction improvements help action selection on the declared targets. | Neither Rope nor Granular has an admissible DINOcular planning outcome. | Replacement depth production is finished; merge and frozen-validate the full caches before admission, a zero-depth reuse decision, or corrected real-depth training from step zero. |
| **Weights & Biases dashboard** | Makes campaign records visible without controlling training. | The dashboard is confirmed, and its currently visible lineage is imported history rather than live telemetry. | Keep imported records distinct from live telemetry as new runs report. |
| **Empirical paper** | Records the design and will eventually report the evidence. | Framing, Methods, the overview figure, Limitations, strict result ingestion, and predeclared decision reporting exist. Partial controller-accepted seed-one Results in Progress values include valid DINOv2 prediction and planning rows, but no aggregate decision, winner, or conclusion exists; excluded or provisional Rope and Granular depth rows cannot support claims. | Generate the tables, figures, and supported conclusion only after the complete fixed evaluation bundle exists. |

## Current milestone

The immediate priority is one complete matched seed across all four tasks,
with DINOv2 versus DINOcular as the main comparison. The existing seed-one
lineages are used because they already contain the most progress; the numeric
seed label has no special scientific status. PushT, Wall, and DINOv2 continue
unchanged. Rope and Granular zero-depth histories remain pending exact
functional reuse proof. Their corrected canaries passed, and full replacement
depth production finished on 30 July 2026 with every shard complete; cache
merge and frozen full-cache validation are now running and must pass before
admission, a zero-depth reuse decision, or corrected real-depth training from
step zero. Seeds two and three
remain preserved until this first all-task comparison is complete.

The next meaningful evidence is a fully trained and evaluated DINOv2–DINOcular
pair on the same task. Completing all four such pairs gives the first
cross-task answer for one matched seed. Until evaluation and planning finish,
training progress alone does not answer the research question.

The later milestone is the complete 36-lineage matrix: all three systems on all
four tasks with all three seeds, followed by the predeclared comparisons,
figures, and empirical paper results.

## Current decisions

- CLAIX/Aachen is reserved for other work and is not available to this project.
  Scientific jobs may use Marvin, lmgpu, or lmdort when the accepted
  environment and inputs can be preserved. lmdort was proven compatible and
  used; Marvin remains the main campaign cluster. lmgpu is used only for
  lineages whose complete accepted environment can be reproduced there.
- All 36 lineages remain controller-managed, with at most one computing job in
  each task-and-system chain and at most 12 project GPUs in use.
- First complete the DINOv2 and DINOcular seed-one pair on every task, using
  parallel capacity across tasks. PushT, Wall, and DINOv2 continue unchanged.
  Rope and Granular corrected canaries passed and their full replacement depth
  production finished on 30 July 2026; cache merge and frozen full-cache
  validation are now running and come before admission, a zero-depth reuse
  decision, or corrected real-depth training from step zero. Their zero-depth
  histories
  await exact functional reuse proof. Seeds two and three remain preserved
  until the first all-task comparison is complete.
- All nine runs for one task stay on one GPU model; resumed runs retain that
  model.
- Each lineage proceeds automatically from training to held-out prediction and
  then fixed planning.
- No scientific claim is made before the fixed evaluations are complete.

## Work meter

Direct effort is the ongoing 36-lineage training campaign and its prepared
automatic prediction and planning evaluations. Support work remains bounded.
No exact productive-work percentage is claimed without measured accounting.
