# DinocularWorldModel — human plan

## Research question

Does giving a robot's world model a sense of depth help it predict what happens
after an action, and plan actions that finish the task?

A world model here is a system that looks at recent camera images, is told what
action the robot takes, and predicts what the scene will look like next. That
prediction is then used to plan: try many candidate action sequences in
imagination, and pick the one whose predicted outcome best matches the goal.

The study asks two versions of the question. The broad one: is a depth-aware
visual system better than a colour-only one? The narrow one: within the same
depth-aware system, does depth that actually carries information beat depth that
carries none?

## Why the design changed on 2026-08-04

The four original tasks were inspected in their own source code. Two of them
cannot have depth at all.

- **PushT** was a flat two-dimensional physics world. Objects had a position and
  a rotation angle and nothing else. The picture was a diagram drawn with vector
  shapes. There was no camera, no viewpoint, no third dimension.
- **Wall** had no physics engine and no picture-making step at all. Its images
  were assembled directly as coloured pixels: white background, black pixels for
  a wall, a soft red dot for the agent. There was no scene for a camera to look
  at.

Depth means "how far away is the thing in this pixel". In those two worlds the
question has no answer. The depth maps previously fed to them were produced by a
depth-estimating network run over flat drawings. The network returns numbers,
but the numbers describe nothing, because there is nothing to describe.

This mattered because the study had made exactly those two depth-free tasks its
primary task and its control, while the two tasks that do have real depth were
labelled exploratory and set aside. The question the project exists to answer
was being asked only where it could not be answered.

Two things followed. The Wall depth release the plan was waiting for can never
arrive, because there is no Wall geometry to release. And two long PushT
training runs, about seventy per cent complete, were spending compute on a
comparison whose depth channel was empty. Both were stopped on 2026-08-04.

## What the depth actually looks like now

Rope and Granular were confirmed genuinely three-dimensional: a particle
simulator, a robot arm, and a camera placed above and off to the side looking
down at forty-five degrees. Their simulator records true depth for every frame.

The important question is not whether depth exists but whether it *changes when
the scene changes*. If depth only ever shows the static floor, it tells the
model nothing about what the robot is doing. This was measured directly, by
comparing an early and a late frame of the same episode and asking how depth
behaves at the pixels where the picture changed:

- **Rope:** every single pixel where the rope moved also changed in depth.
  Pixels where nothing moved changed by exactly zero.
- **Granular:** of the pixels where the material moved, 84 per cent changed in
  depth. Moving regions changed roughly six thousand times more than still
  regions.
- **OGBench-Cube:** the strongest of the three. About a sixth of the picture
  changes, and depth over those regions changes roughly fifteen thousand times
  more than over still regions.
- **The estimated depth that was rejected earlier** scored 1.4 on Rope and 1.6
  on Granular by the same kind of comparison — barely distinguishable from no
  signal at all.

Every task admitted to the study has now passed this same test, measured the
same way. Nothing is admitted on the assumption that it "looks three
dimensional".

So the earlier decision to throw out the estimated depth was correct, and the
real simulator depth is enormously better. This is the strongest indication so
far that the project's question can be answered at all.

A caution about how this looks: in a picture of the depth map, the rope and the
granular material are nearly invisible, because the colour scale has to cover a
floor that recedes to twenty-four metres while the objects move by centimetres.
The signal is real; it is simply too fine to see against that range. The
measurement, not the picture, is the evidence.

## The comparison

Four systems:

- **DINOv2:** a colour-image system with no depth input.
- **DINOcular:** a depth-aware system given colour and real simulator depth.
- **Zero-depth DINOcular:** the same depth-aware system with identical internals,
  but every depth value replaced by zero.
- **Shuffled-depth DINOcular:** the same system given real depth belonging to a
  *different* frame.

The fourth system was added on 2026-08-04 to close the study's largest remaining
weakness. The zero-depth system is not an ordinary colour-only model: an
all-zero depth map is unlike anything it saw while being built, so if it does
worse, that could reflect the unfamiliar input rather than the missing
information. Those two explanations could not be told apart.

Shuffled depth separates them. It supplies genuine depth, with entirely normal
appearance and statistics, that simply belongs to the wrong moment. The input
stays familiar; only the correspondence between what is seen and how far away it
is gets destroyed. If depth genuinely carries useful information, this system
should lose ground against real depth. If a zero-depth drop were only an
unfamiliar-input effect, shuffled depth should hold up.

The substitute is taken from a different moment of the *same* episode, chosen
from a fixed seed, so a run reproduces exactly and no frame ever keeps its own
depth. This detail matters: an earlier version drew the substitute from whatever
else happened to be alongside it during training, which usually meant a
different episode entirely. That would have destroyed two things at once — the
timing and the scene — and the arm exists to isolate only the first.

An outside reviewer raised the opposite worry: if most of a depth map is the
unchanging table and floor, then swapping in another moment of the same episode
might barely change anything, and the comparison would be too gentle to show
anything. This was measured on 30 episodes of each task, separating pixels where
the scene moves from pixels where it does not.

The worry turns out to be unfounded, and the same-episode choice is the stronger
one. Where nothing moves, the substituted depth is almost unchanged, as intended.
Where things move, it changes a great deal — 29 times more than the still regions
on rope, 25 times more on granular, 6 times more on the cube task. And on all
three tasks the same-episode substitute disturbs the moving regions *more* than
a substitute taken from another episode does: 25.0 against 15.0 on rope, 88.4
against 80.7 on granular, 2,177 against 1,963 on the cube. The reason is simple:
five steps later in the same episode the object has genuinely moved somewhere
else, whereas a random other episode may happen to leave it in a similar place.

So the comparison damages exactly what it is meant to damage, leaves the rest
intact, and does so more thoroughly than the alternative.

## The tasks

**Committed, with real depth confirmed by measurement:**

- **Rope** — a deformable string pushed by a robot arm. 1,000 recorded runs of
  20 frames, 20,000 frames, real depth for every one.
- **Granular** — a pile of loose material scattered by a robot arm. Same size,
  same coverage, and the stronger depth signal of the two.

- **OGBench-Cube** — a three-dimensional pick-and-place task from an existing
  public benchmark, where blocks are lifted clear of the table. It was put
  through the same measurement as the other two before being accepted, and it
  produced the strongest depth signal of any task in the study: about a sixth of
  the picture changes as the scene evolves, and depth over those moving regions
  changes roughly fifteen thousand times more than over still regions. It is
  also the task the two most credible competing systems report on, so results
  land directly alongside theirs.

  Two things had to be established rather than assumed. Its own recorded
  datasets store robot and object coordinates, not pictures, so image and depth
  data must be generated rather than downloaded. And the depth must be rendered
  deliberately, since it is not part of the default observation.

  A first attempt at generating that data failed in a way worth recording. The
  robot was driven by random actions, which never actually touch the block. The
  result looked complete — a thousand episodes, correct image and depth sizes,
  depth that changed from frame to frame — but the block never moved in any of
  them, and the robot's own position was never written down. A model trained on
  it would have learned only how the arm swings, and would then have been asked
  to plan towards block positions it had never seen change. The depth did vary,
  but that variation was the arm passing through view.

  The data was regenerated by replaying the benchmark's own recorded expert
  behaviour, which does move the block. Every episode checked now shows the
  block travelling a real distance, and the robot's position is recorded. A task
  now has to pass a second admission test alongside the depth one: **the thing
  the robot is supposed to manipulate must actually move in the recorded data.**

**Built and verified, now optional:**

- **PushT-3D** — a genuine three-dimensional rebuild of the push-T task on a
  table with a robot and a real camera. Its depth is real but modest: the
  T-shaped block stands about five and a half centimetres above the table, far
  above the simulator's precision but far weaker than the other three tasks. It
  also requires its data to be generated, exactly like OGBench-Cube, so it
  offers no saving in effort. It is kept only as optional continuity with the
  original benchmark and does not carry any comparison.

**Retired:** Wall, and the flat two-dimensional PushT.

## The training runs

One run means training one system on one task with one random seed. Four
systems, three tasks, three seeds gives **36 runs** — nine matched sets of four,
one per task and seed. Within a set everything is identical except the visual
system: same data, same ordering, same starting point, same length. Comparisons
are made inside a set first, then summarised across seeds.

The three tasks are Rope, Granular and OGBench-Cube. All three now sit in the
run configuration; the retired flat push-T entry is gone and OGBench-Cube's
pictures and depth exist on the cluster.

**The first seed is fully trained — all twelve runs.** Rope and Granular
finished their three depth-aware systems on 7 August, OGBench-Cube finished all
four of its systems on 10 August, and the two colour-only Rope and Granular runs
were already finished from earlier. Every one reached the same fixed length of
53,500 updates and 100 passes over its data, so the matched sets really are
matched. Twelve of the thirty-six runs are therefore done, and they are exactly
the twelve that make up one complete comparison.

The remaining twenty-four are the second and third seeds. They have not been
started.

## How one run is trained

The pretrained visual system is frozen — the study does not retrain DINOv2 or
DINOcular. It trains a fresh predictor and fresh action and state modules for
each system, task and seed:

1. Take a short ordered window from a recorded run.
2. Turn each image, and its depth where applicable, into visual features using
   the frozen system.
3. Give the predictor those features plus the recorded action and robot state.
4. Train it to predict the features that should come next.
5. Repeat in fixed order until the task's exact update count is reached.

Reconstruction of the raw picture is switched off, so the model is trained and
judged in feature space. Every run saves enough to resume exactly after an
interruption. After training, each model predicts held-out runs and then plans
towards fixed goals. Goals never tune the model.

## How the results will answer the question

**Is the depth-aware system better?** Planning success is the primary measure.
Prediction error is supporting context only: the two systems predict in
different internal spaces, so their error numbers are not directly comparable,
and no conclusion rests on them.

**Does informative depth help?** Judged on Rope and Granular, where depth
demonstrably tracks what moves.

Uncertainty comes from repeated resampling of seeds and goals. Three seeds test
whether the effect is consistent across these particular trained models. They
are too few to speak about every possible future seed, and the write-up will say
so plainly rather than implying more.

**How rope and granular are scored.** Success needs a rule for when a planned
result is close enough to the goal. Two candidate rules were compared on 40 real
episodes of each task: the distance the study has been using, which ignores which
particle is which, and a second one that keeps the particles in order.

Keeping the order is clearly better. On granular it separates perfectly — every
near-goal configuration falls below a cutoff that no unrelated configuration
reaches. On rope it admits 92 per cent of near-goal shapes while wrongly
admitting 12 per cent of unrelated ones, against 35 per cent wrongly admitted by
the order-ignoring distance. Rope is still the weaker of the two, but it is no
longer hopeless, and the earlier conclusion that rope could not have a pass/fail
score at all was an artefact of the weaker distance.

One honest caveat: "near the goal" was stood in for by an earlier frame of the
same episode. The closer that frame is to the end, the better both rules look,
and separation falls away to nothing by fifteen steps back. That is expected — a
frame fifteen steps earlier really is a different configuration — but it means
the cutoff describes short-range accuracy, not any distance at all.

**The success rules are being rewritten and are not yet fixed.** The previous
rule required six separate conditions to hold at once, which made a false
negative far more likely than a true one, and one of its required conditions was
a measure the same document described as invalid for that comparison.

An outside reviewer proposed a cleaner replacement: work out the improvement
seed by seed, build an uncertainty range from those few numbers, claim depth
helps if the whole range sits above ten points, claim it does not help if the
whole range sits below five, and report anything else as undecided.

That rule was then simulated before adopting it, and it does not survive the
test. If the true improvement is exactly the ten points the rule asks for, the
rule announces a positive result only twelve times in a hundred. It needs a real
improvement of about twenty-five points before it fires reliably. And when depth
genuinely makes no difference at all, it returns the correct "no effect" verdict
only about four times in ten — the rest of the time it says undecided.

So the rule is honest but far too demanding, and adopting it as written would
have built a study that mostly cannot conclude anything. It needs its bar
lowered, or the improvement needs to be estimated across seeds and goals
together rather than from three numbers. This is the one item that still needs a
decision rather than more work.

The same simulation answered a second question. The reviewer suggested dropping
the zero-depth system and spending those runs on a fourth seed instead. A fourth
seed is worth between two and six points of extra reliability, depending on how
large the real effect is — and nothing at all when there is no effect to find.
That is too little to pay for losing a whole comparison, so the four systems
stay.

## What we know now

Rope and Granular have real depth across all training frames, all validation
frames, the fixed prediction protocol, and every planning start. Their
colour-only runs for the first seed are fully trained with 100 held-out
prediction episodes each; those results are preserved and still valid. Their
depth-aware runs have now been retrained against the real thing and are finished,
as have all four OGBench-Cube runs — the whole first seed, twelve runs, exists.

The Rope real-depth preparation was blocked by a small bug: a file-path naming
mismatch between how the depth records were written and how the checking step
expected to find them. The 20,000 depth frames themselves were complete and
correct all along — only the confirmation step failed, which then made a second
check report Rope as unusable. Both were the same one-line problem. It is fixed
and the preparation finished successfully in nine and a half minutes.

The success measure for Rope and Granular was broken: it asked whether a
distance was less than zero, which no distance ever is, so every planning
attempt was recorded as a failure whatever actually happened. This is now fixed.
The cutoffs it compares against are no longer placeholders: rope and granular
both carry their measured order-keeping values, and the scoring code has been
confirmed to use the order-keeping distance rather than the order-ignoring one.

A second and larger fault in the planning setup was found on 2026-08-07 and
fixed. The planner had no limit on how long it would keep trying. It was set to
stop only once *every* goal had succeeded — which, with a hundred goals, never
happens. A planning run would have carried on until the cluster cut it off at
its time limit, and because results are saved only once planning finishes, it
would have saved nothing at all. Every planning job would have come back empty.
The planner now takes exactly one pass of five steps: the same horizon the goal
itself is built from, so no system is handed more time than the goal allows.

Their planning evaluation used ten goals, too few to support a claim. It is now
one hundred. Fifty was the original replacement, but a check of how precisely
this study can measure a difference showed that fifty goals across three seeds
leaves the intended threshold — a ten-point improvement in the share of tasks
completed — sitting right at the edge of what could be distinguished from noise.
The study could have missed a real effect and reported nothing. Doubling the
goals buys that precision back, and costs only evaluation time, not training.

PushT-3D exists and works: the simulator runs locally, delivers real camera
depth, and the environment is built and checked.

We now also know what the two visual systems were built from, which had never
been written down. The colour-only system's own documentation names its training
set only by a label and refers the reader to a paper, so what is inside it cannot
be established from anything we hold. The depth-aware system was built from real
multi-view photographs and ImageNet pictures, with its depth supplied by a
depth-estimating network rather than measured.

Nothing on record suggests either system ever saw simulator renderings, so the
worry that a good result could be memorisation rather than genuine understanding
has no support — though for the colour-only system that is an absence of
evidence rather than a clean answer.

The more interesting consequence is the opposite one. The depth-aware system was
built on *estimated* depth over ordinary photographs, and this study feeds it
flawless simulator depth. That is unfamiliar input in its own right, which is
precisely the effect the zero-depth and shuffled-depth comparisons exist to
separate — and another reason not to drop either of them.

No matched comparison, aggregate result, winner or conclusion exists yet. What
does exist, for the first time, is a working path from a trained model to a
planning score for both kinds of system — colour-only and depth-aware alike —
and one complete first seed of trained models waiting for that path to be run at
full scale.

## Where this sits in the field

A check of recent work confirms the project is not scooped, and turned up
something useful.

The most visible recent competitor is a system from a well-known group that
plans far faster than the approach this study builds on. It is a fair
comparison point. But it is still evaluated on the *flat two-dimensional*
push-T and a flat two-dimensional navigation task. The one paper that looked
like a direct rival on depth turns out to use depth only while training, never
at use time, and never evaluates planning at all — so the specific question here,
depth supplied at the moment of use and judged by whether the robot finishes the
task, is unoccupied.

That has a consequence worth stating: the benchmark suite this subfield uses for
these models is dominated by tasks in which depth cannot exist. The observation
that the question cannot be asked there is not only a repair to this project. It
is a finding about how these systems are being evaluated generally, and it
belongs in the write-up.

## Current milestone

One complete matched seed on Rope and Granular using real depth, followed by
their planning evaluation. These two tasks can deliver the project's first
genuine depth result without anything new being built.

All four items on this list are now done. What remains is running the evaluation
itself, which is waiting for machines.

1. **Settled.** The success rule is fixed: pool the individual goal outcomes
   from every seed into one collection, then resample that collection to get a
   95 per cent range for the difference between two systems. This avoids the
   trap the earlier candidate fell into, where a verdict had to be squeezed out
   of only three numbers and so almost always came back "undecided".
2. **Done.** Rope and granular score with the order-keeping distance and their
   measured cutoffs, both confirmed in the code rather than assumed.
3. **Done.** The cube task points at its regenerated data. Its depth is wired
   through the same checked path as rope and granular, confirmed by a short test
   run in which the depth genuinely reached the visual system.
4. **Done.** The six rope and granular depth-aware runs finished retraining
   against real depth on 7 August, and the four cube runs finished on 10 August.
   All twelve first-seed models exist.

Getting from a trained model to an actual planning score turned out to need
several repairs, each of which would have produced nothing or nonsense:

- The planning code could not read the files training saves. It expected the
  models packaged one way; training writes them another. No trained model could
  have been loaded at all. It now reads both, checked by comparing all 950
  stored weights against the file.
- Planning also could not start, because loading it dragged in a retired task
  whose physics library is not installed. That task is not part of the study and
  has been dropped from the list of environments.
- The rope and granular simulator crashes while building its fourth copy in one
  process, and planning was building one per goal — a hundred of them. It now
  uses a single simulator, which is safe because the goals are already handled
  one after another, each starting from its own reset.

The cube's two missing software packages are now installed alongside the
prepared environment rather than inside it, and were checked on a graphics card
to confirm they sit beside the existing simulator and learning libraries without
displacing any of them. One of them arrived with its own copy of a core
numerical library, which replaced the container's and broke the learning library
outright; that copy was removed and the check repeated.

With all of that in place, one complete planning attempt was run end to end for
the first time, and it produced a real score: a rope goal reached, at a distance
of 0.48 against the measured cutoff of 1.13. That is the first time anything in
this project has travelled from a trained model to an actual planning result.

Measuring it suggested the cost question was settled. One goal took about 80
seconds to set up and about 65 seconds to plan and score, so a hundred goals
should take roughly four hours against a job limit of just under eight. That
estimate turned out to be wrong in practice, and finding out cost a day.

**The obstacle that blocked the whole comparison is now gone.** Until 7 August
the depth-aware systems could not be planned with at all. Depth came from
recorded files that accompany each saved run, and planning does not replay
recorded runs — it steers the simulator into situations that were never
recorded — so there was no stored depth to look up.

The repair landed and has been checked three ways. The simulator's drawing
routine returns five channels, of which the code was keeping three and throwing
the depth away; it now keeps it. That live depth is then converted into exactly
the same form the recorded training files use, and the conversion was compared
line for line against the program that produced the training depth: it is the
identical formula, reading the same per-task constants from the same stored
description of the task. On rope, the depth the live simulator produces spans
9.51 to 24.06 metres against the recorded description's 24.064 — they agree.
This mattered because depth scaled even slightly wrong would look entirely
normal while quietly corrupting the study's central comparison.

The third check is the one that counts: depth-aware planning has actually run
and written scores. The system that gets real depth and the system that gets
depth from the wrong moment have both produced planning results on rope, and
real depth has produced one on granular too. Nothing is being taken on trust
from the code.

**A risk that came with that repair, now checked and cleared.** Once planning
takes depth live from the simulator, an obvious worry appears: the two control
systems are defined by the depth they are *denied*, and the live simulator hands
out real depth to everybody. If the denial only happened while reading recorded
files, both controls would quietly become the real-depth system the moment
planning started, and the study's central comparison would be empty while
looking perfectly healthy.

It does not happen. The denial is built into the visual system itself, not into
the file reading: the zero-depth system blanks the depth as it enters the
encoder, and the wrong-moment system substitutes it there. Every observation
passes through that same door — recorded or live, goal or rollout. So the two
controls stay controls while planning, and this was established by reading where
the branch actually sits, not by assuming the arrangement carried over.

**What went wrong on 8 August, and what it cost.** All eight full evaluation
runs — the ones that take a trained model and try a hundred goals — were
launched and all eight died at the eight-hour limit. That alone would have been
survivable, except for how the program saved its work: it wrote every score at
the very end, after all hundred goals. A run that finished ninety goals wrote
down nothing. Eight runs, days of machine time, no saved scores. The runs were
also silent for their last five hours, so there was no way to tell whether they
were merely slow or stuck.

The repair was to save as the run goes: the goals are now worked through in
batches of ten, and the scores for each batch are written to disk immediately.
A run cut off now loses at most nine goals' work instead of all hundred. Each
batch also prints how long it took, so a stall shows its exact location instead
of five silent hours. This was proved rather than assumed — a small run was
deliberately cut short by its time limit, and the goals it had already finished
were sitting safely on disk with correct scores. The runs also moved to a
day-long queue instead of the eight-hour one.

Why they needed more than eight hours is still open. The new batch timings will
answer it on the next attempt.

**What the first scores actually say — very little, so far.** Four systems have
each produced planning scores on one or two goals: colour-only rope, real-depth
rope, wrong-moment-depth rope, and real-depth granular. Every attempt succeeded.
The distances are 0.48, 0.48 and 0.49 on rope against a cutoff of 1.13, and 1.08
on granular against a cutoff of 1.77.

Two things follow. The machinery works end to end for both the colour-only and
the depth-aware systems, which is the real news. But no comparison can be read
from these numbers: one or two goals per system is nothing, and the fact that
everything passed was a warning.

**That warning has now been measured, and it is serious.** The hundred goals were
reconstructed exactly as the evaluation builds them, and the distance from each
goal's starting position to its target was compared against the same cutoff the
evaluation uses. The result: on rope, 39 of the 100 goals already sit inside the
cutoff before the planner does anything at all. On granular, 82 of 100 do.

Those goals are free passes. No system can fail them, and no system can be
distinguished by them. On granular that leaves 18 goals out of a hundred doing
any real work, which is far too few to support the comparison the study exists to
make. Had the eight queued evaluations run as they stand, they would have
returned four nearly identical scores and answered nothing.

The cause is simple and fixable: a goal is built by taking a recorded moment and
looking five steps ahead. Five steps is not far enough for the scene to change
much, so start and target are often already the same thing by the measure's own
standard. The repair is to build goals further ahead, and a sweep over longer
look-aheads is running now to find the shortest one that makes the goals
genuinely hard. Nothing about the metric, the cutoff, or the success rule
changes — those were measured honestly and stay as they are. Only the goals get
harder.

This is the right kind of failure to find before spending the machine time, not
after.

**Where the work is stuck right now: the queue.** All eight full evaluation runs
are submitted and waiting. The cluster currently estimates they will not begin
until 15 August — four days of waiting, on the machines that answer the study's
question. This is not a fault in the study; it is contention with other users.
Asking for a shorter time limit does not move them forward. The one lever that
might is a second, much emptier group of machines on the same cluster, whose
graphics cards are a different and slightly older model. Whether the runs work
there at all is being tested now with a short probe.

**OGBench-Cube is trained but has never been planned with, and now we know why.**
Its four first-seed models are finished, but a cheap two-goal probe failed in
twelve seconds: the cube has no planning settings file at all. Rope and granular
each have one; the cube was never given one when it joined the study. That is a
small, contained gap and it is being filled now. Until it is filled the cube
contributes training only.

One measure has had to be dropped. Prediction error was only ever supporting
context, and it is now clear it cannot be produced for these runs at all: the
machinery that computes it is tied to a record-keeping apparatus these runs were
deliberately launched without, and it refuses to run without it. This costs the
study little. The two visual systems predict in different internal spaces, so
their prediction errors were never comparable to each other in the first place,
and no conclusion was ever going to rest on them. Planning success now stands
alone as the measure, which is what the study said it would be judged on.

The later milestone is the full matrix, then the comparisons, figures and paper.
Training progress alone does not answer the research question.

## The next useful results, in order

1. **Make the goals hard enough to be worth running.** This now comes first,
   because the eight queued evaluations would otherwise answer nothing. A sweep
   over longer look-aheads is running; it names the shortest one that leaves
   fewer than fifteen free passes in a hundred on both tasks. Then the eight runs
   are resubmitted against those goals.
2. **Give the cube its planning settings.** A one-file gap, being filled and
   re-tested now. If it works, four cube evaluations join the queue and the study
   gains its strongest-depth task.
3. **Find out whether the second group of machines can be used.** The eight runs
   currently wait until 15 August on the congested machines. A probe on the
   less-busy ones is running. If it passes, the same work starts days earlier.
4. **The second and third seeds start training.** Twenty-four runs. They are
   worth queueing only once the first seed's evaluation is known to produce
   numbers that can separate the systems. Starting them sooner risks training
   twenty-four models against a measure that cannot tell them apart.

The first item is the one that decides whether the study can conclude anything.
The rest remove waiting.

## Current decisions

- Every task in the study must supply real simulator depth. No task uses
  estimated or substituted depth.
- Wall and the flat two-dimensional PushT are retired. Wall has no scene, so no
  depth release for it will ever exist. The two in-progress flat PushT runs were
  stopped on 2026-08-04.
- Rope and Granular carry the informative-depth question, because their depth
  demonstrably tracks what moves.
- PushT-3D is kept for continuity but does not carry the decisive comparison.
- OGBench-Cube is adopted only if its depth is verified the same way Rope and
  Granular were — by measurement, not assumption. Its depth passed that test and
  its first-seed models are trained, but it only joins the comparison once its
  planning is shown to run.
- Planning runs save their scores as they go, in batches of ten goals, so a run
  cut off by the cluster keeps everything it finished. Saving only at the end
  once cost eight full runs.
- Depth used at planning time is converted with the identical formula and the
  identical per-task constants as the recorded training depth. This was checked
  against the program that produced the training files, not assumed.
- Depth-aware runs are retrained against real depth, never relabelled.
- The success cutoffs are measured from data and now fixed in the configuration
  for both rope and granular. No placeholder remains.
- Each planning attempt gets exactly one pass of five steps, the same horizon
  the goal is built from. Letting the planner keep retrying would both hand it
  more time than the goal allows and, as written, never stop at all.
- Rope and granular are scored with the order-keeping distance, not the
  order-ignoring one, because it separates measurably better on both tasks.
- All four systems stay. A fourth seed in their place would buy at most a few
  points of reliability, and the depth-aware system was built on estimated rather
  than measured depth, which makes both the zero-depth and the shuffled-depth
  comparisons more necessary, not less.
- The substituted depth keeps coming from the same episode. It was measured to
  disturb the moving parts of the scene more than a substitute from another
  episode would, while leaving the still parts alone.
- The success rule is settled and closed. Every goal outcome from every seed
  goes into one pooled collection, which is resampled to give a 95 per cent
  range for the difference between two systems. The earlier candidate, which
  built its range from three per-seed numbers, was simulated and rejected: it
  returned "undecided" most of the time even when a real effect was present.
- Prediction error is not reported. It was supporting context only, it is not
  comparable between the two visual systems, and it cannot be produced for these
  runs without record-keeping they were not launched with.
- Outside work is only cited if it has released code and more than one author.
- No claim is made before the fixed evaluations and the rewritten rules support
  it.

## Work meter

Direct effort is the move to real-depth tasks and the training and evaluation
that follows. No exact percentage is claimed without measured accounting.
