# DinocularWorldModel restart handoff

Status: CAN RESTART
Last updated: 2026-09-01

## Goal

Does depth supplied at the moment of use improve world-model prediction and
robot planning? Judged by planning success across four visual systems
(dino_pinned, dinocular, dinocular_zerodepth, dinocular_shuffleddepth) on three
tasks (rope, granular, ogbench_cube), three seeds — 36 runs.

## Where the real work lives

- Cluster project root:
  `/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm`
- Cluster code (authoritative): `<root>/code/dino_wm`, HEAD `4f865ff`
- Local `~/dinocular-wm/dino_wm` is **stale and divergent**. Do not treat it as
  the source of truth.
- Docs repo: `~/dinocular-wm-worktrees/project-docs`, branch
  `task/human-plan-current-update`.

## Training — seeds 1 and 2 complete, 24 of 36 runs

All twelve first-seed runs at step 53,500 / 100 epochs.

- rope + granular, dinocular / zerodepth / shuffleddepth:
  `<root>/outputs/seed1-matched-20260806/{rope,granular}-<arm>-s1`
- rope + granular, dino_pinned:
  `<root>/outputs/campaign-seed1/rgb-dino-seed1-20260728a/{rope,granular}/run`
- ogbench_cube, all four arms:
  `<root>/outputs/matched-seeds-20260807/ogbench_cube-<arm>-s1`

**Seed 2 is also complete** — all twelve runs at step 53,500, finished
2026-08-17, in `<root>/outputs/matched-seeds-seed23/<task>-<arm>-s2`. They were
launched with `tools/matched_run_seed23.sbatch`. 24 of 36 runs trained.

Seed 3 was cancelled deliberately, not lost. The paired bootstrap resamples
*goals*, so precision is bought far more cheaply by adding goals than by adding
seeds; a second seed only answers "did one trained model happen to ignore
depth", and one extra seed answers that as well as two.

## The goal-difficulty repair — works, and changes the cost model

Rejecting already-solved goals at sampling time (uncommitted in `plan.py`,
being committed by the sharding work) removes every free pass:

- rope 39/100 → **0/100** below the 1.132036 cutoff (min distance 1.1599)
- granular 82/100 → **0/100** below 1.770630 (min 1.7713)

**The hardened goals discriminate.** A real rope/dinocular run scored one goal
solved at ordered distance 0.484 and one failed at 2.057, success rate 0.5.
Before the repair every goal passed.

**Two earlier jobs that "hung" were never hanging.** They were simply too slow
for their walltime. Instrumentation showed goal sampling costs milliseconds
(15 ms per dataset load, 5–10 ms per triviality check). The cost is in the
simulator: a trivial goal is one where the rope barely moved, so its replayed
pushes were ~60 sim substeps; a real goal has real pushes of 1200–1560
substeps. The old 84-second figure for two goals was itself an artefact of
scoring goals where nothing happened.

**Measured real cost: roughly 10–15 minutes per goal** (41 rollout calls for
2 goals in 18 minutes). 100 goals is about 20 hours per run, so one job per run
is not viable. Goal sharding (`goal_shard_start`, `goal_shard_count`) was added
to `plan.py` for this: the full 100-goal list is sampled with the existing
seeded RNG and then sliced, so shard *k* holds identical goals for all four
systems, and every per-goal record carries its global index.

## OGBench-Cube — both defects fixed, now fully scored

Its goals are sound: only 2 of 100 are free passes (median start-to-goal
distance 0.124 m against a 0.04 m cutoff), far better than rope or granular.
Two defects were found and both are now fixed.

1. `4f865ff`. The live wrapper emitted no `proprio`. The four cube models were
   trained on a constant-zero length-1 proprio placeholder (dataset
   `proprio_dim=1`, checkpoint `Conv1d in_chans=1`), so the wrapper now emits
   exactly that. Note for the write-up: **the cube models carry no real
   proprioceptive input.**
2. `ef7d8d6`. A 2-goal smoke wrote `state_dist [0.0, 0.0]`, `success
   [1.0, 1.0]`. Exact zeros are not measurements. `set_state` wrote **only**
   `object_joint_0.qpos[:3]` — the block — and never the arm, so the arm never
   contacted the block, the block's final position was decided purely by
   settling and was therefore independent of the actions, and goal rollout and
   evaluation rollout landed bit-identically. The full joint state (21-dim
   qpos, 20-dim qvel) was recovered from `cube-single-play-v0.npz` and is now
   restored before replay.

All four cube arms have since scored 100 real goals each.

## The planner was replaying the simulator 30 times per goal

`conf/plan_*.yaml` had CEM `opt_steps: 30` with `eval_every: 1`. CEM optimises
entirely through the world model on GPU (`planning/cem.py:104-108`); the
simulator is touched only by `evaluator.eval_actions` at `cem.py:123`, which
exists for logging and additionally breaks the loop early once a goal already
succeeds (`cem.py:131`). So `eval_every: 1` replayed PyFleX ~30 times per goal.

One rope goal cost **over 45 minutes** and two probe jobs died without finishing
a single goal. With `eval_every: 31` (above `opt_steps`, so no in-loop
evaluation at all) the same goal costs **57.8 s** and scores 0.490 where the
slow setting scored 0.484. Committed as `3965d42` for all three tasks. Every
system now gets the full 30 CEM steps and is scored once.

## All four systems receive identical goals — verified

The goal draw uses python's `random`, and `seed(cfg.seed)` runs *before* the
dataset build and model load, so a different architecture could in principle
have shifted the goal sequence and voided the matched design.

Checked directly (`scratch/rng_identity.py`, jobs 26974625 and 26974630): after
the dataset build and after the model load, all four arms — including the
colour-only baseline with its different encoder — report python RNG state
`1b09c9baf493` and identical next draws `[413, 389, 204, 613, 183, 235]`.
So the goal list is identical across arms, and the paired analysis is valid.

## Both seeds are fully evaluated: 24 runs, 200 paired goals per task

All twelve seed-2 evaluations completed 2026-08-26, as did the rope
shuffled-depth top-up. Every arm now has 100 goals per seed.

| task | seed | colour-only | real depth | zero depth | wrong-moment depth |
|---|---|---|---|---|---|
| rope | 1 | 0.44 | 0.50 | 0.46 | 0.15 |
| rope | 2 | 0.50 | 0.53 | 0.52 | 0.13 |
| granular | 1 | 0.60 | 0.60 | 0.55 | 0.02 |
| granular | 2 | 0.64 | 0.68 | 0.52 | 0.07 |
| cube | 1 | 0.99 | 0.97 | 0.98 | 0.13 |
| cube | 2 | 0.97 | 1.00 | 0.98 | 0.16 |

Paired bootstrap over shared goals, both seeds pooled, n=200
(`tools/aggregate_planning_pairs.py`):

| comparison | rope | granular | cube |
|---|---|---|---|
| depth − colour | +4.5 [−3.5, +12.5] | +2.0 [−4.5, +8.5] | +0.5 [−1.5, +2.5] |
| **depth − zero depth** | +2.5 [−5.5, +10.5] | **+10.5 [+3.0, +17.5]** | −0.5 [−1.5, +3.0] |
| wrong-moment − depth | −37.5 [−45.0, −30.0] | −59.5 [−66.5, −52.0] | −84.0 [−89.0, −78.5] |

**The broad question comes back negative, and now with a usable bound.**
A depth-aware system does not beat a colour-only one. Granular and cube exclude
the study's own 10-point threshold outright; rope's upper limit is +12.5, still
just above it.

**The narrow question comes back positive on granular.** Real depth beats
uninformative zero depth by +10.5 pp, interval clear of zero. This is the
comparison rope and granular were admitted to carry, and granular carries it.

One honest caveat: the granular per-seed gaps are −5 and −16, so the pooled
significance leans on seed 2. That is between-seed variance, which more goals
cannot reduce — only a third seed could. Rope shows nothing (+2.5), and the
cube is at a 98% ceiling where nothing can be separated.

## The trained real-depth model does use depth

The plan-time ablation smoke settles the interpretation that was open. Take the
model *trained* on real depth and withhold depth at planning time only, on the
same three rope goals:

| goal | real depth | depth withheld |
|---|---|---|
| 0 | 0.483 (pass) | 3.178 (fail) |
| 1 | 2.228 (fail) | 2.603 (fail) |
| 2 | 0.211 (pass) | 1.705 (fail) |

2/3 becomes 0/3 and the distances blow up six- to eightfold. So the model is
not ignoring depth — it genuinely relies on it once trained with it.

That kills the easy explanation for the parity and leaves the stronger claim:
the zero-depth arm, trained without depth, learns an RGB-only solution that is
just as good on rope and the cube. Depth is used when supplied, but on those
two tasks it buys nothing that RGB does not already provide. Granular is the
exception where it does.

Full 100-goal ablations for all three tasks are queued as `27232597`-`27232599`.

Together with the three earlier measurements — the encoder responds to depth as
strongly as to RGB (0.162 vs 0.138 on rope); depth carries geometry RGB cannot
predict (full-resolution residual R² 0.35 to −1.32 on rope, 0.02 to 0.58 on
granular); and the 32×32 pooled probe's R² ~0.95 was a smoothing artefact that
must not be cited — the picture is coherent and is a real finding rather than a
broken benchmark. The measure separates systems whenever there is something to
separate: 0.00 → 0.14 → 0.50 → 0.98.

Camera geometry is not the explanation. All four PyFleX rope/granular cameras
sit at the same 45-degree elevation, differing only in azimuth
(`env/deformable_env/src/sim/sim_env/cameras.py`), so the original four-camera
probe was one viewpoint tested four times. OGBench-Cube renders from
`cube_env.py:539` 'front' at 20 degrees — already near-grazing — and shows the
same parity. (`scene_env.py`'s 38.8-degree 'front' belongs to the *scene* task,
not the cube.) A grazing-camera regeneration was proposed and cancelled.

## What is running

Three full 100-goal plan-time depth ablations, `27232597` (rope), `27232598`
(granular), `27232599` (cube), `mlgpu_medium`, `--time=23:55:00`.

Seed-1 and seed-2 evaluation timings, for sizing: rope 6:51-7:59, granular
10:10-14:32, cube 1:13-1:32.

## The plan-time depth ablation

The diagnostic that turns the null into a finding, and it needs no retraining.
Take the trained *real-depth* model and withhold depth at planning time only.
If it still plans, that model never used depth and the parity is fully
explained. If it collapses, it did use depth — and zero-depth's equal score then
means an RGB-only solution is simply as good, which is the stronger claim.

`dinocular_zerodepth.yaml` differs from `dinocular.yaml` by exactly one key,
`neutralize_depth_at_encoder_input`, and plan.py rebuilds the model from the
run directory's own `hydra.yaml`. So the ablation is a directory holding a
symlink to the real run's checkpoints and a copy of its `hydra.yaml` with that
one flag flipped — nothing retrained, and the real run never written to. Built
for all three tasks as `<task>-dinocular-s1-planzero`, alongside a new
`CKPT_DIR_OVERRIDE` in `tools/plan_run.sbatch`.

**Always pass `PLAN_TAG` with `CKPT_DIR_OVERRIDE`.** The output directory is
derived from the *parent* of the checkpoint directory, so without a tag the
ablation would overwrite the real run's scores.

## Exact next action

1. Read `27232597`-`27232599` when they finish. The full ablation either
   confirms the 3-goal smoke at scale or overturns it; the whole interpretation
   above rests on it.
2. Decide on seed 3 for **granular only**. Its zero-depth effect is the study's
   one positive result and its per-seed gaps are −5 and −16, so a third seed
   would settle whether +10.5 pp is real or a seed-2 artefact. Four runs, not
   twelve. Goals cannot substitute here: this is between-seed variance.
3. Rope and cube need neither more seeds nor more goals. Cube is at a 98%
   ceiling; rope's depth−zero interval is centred near zero.

## Repository state

- `code/dino_wm` on marvin at `256cca8`. Recent: `d9d8bc4` goal rejection plus
  sharding, `3965d42` eval_every, `256cca8` aggregation fixes, `4f865ff` cube
  proprio.
- Untracked helper and probe scripts remain in `code/dino_wm` and `scratch/`.

## Notes

- DeepSeek workers run through `claix-deepseek` against the self-hosted
  endpoint; the prepaid API balance is negative.
- The cube env runs behind a multiprocess vector env (`env/venv.py`). Printing
  inside it kills the worker with `EOFError`; diagnose the cube with standalone
  single-process scripts instead.
- Prediction error is not reported. Planning success is the single measure.
