# DinocularWorldModel restart handoff

Status: CAN RESTART
Last updated: 2026-08-26

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

## Training — unchanged, first seed complete

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

## OGBench-Cube — planning runs, but its scores are meaningless

Its goals are sound: only 2 of 100 are free passes (median start-to-goal
distance 0.124 m against a 0.04 m cutoff), far better than rope or granular.

Two defects were found and only the first is fixed:

1. **Fixed, committed `4f865ff`.** The live wrapper emitted no `proprio`. The
   four cube models were trained on a constant-zero length-1 proprio
   placeholder (dataset `proprio_dim=1`, checkpoint `Conv1d in_chans=1`), so
   the wrapper now emits exactly that. Note for the write-up: **the cube models
   carry no real proprioceptive input.**
2. **Open, blocking.** A 2-goal smoke wrote `state_dist [0.0, 0.0]`,
   `success [1.0, 1.0]`. Exact zeros are not measurements. Cause established by
   direct experiment (job 26972557, `<root>/scratch/probe_cube_move.py`):
   `set_state` writes **only** `object_joint_0.qpos[:3]` — the block — and never
   the arm. Replaying recorded actions leaves the block bit-identical in x and y
   across all 15 steps; it only falls ~0.10 m in z and stops. Recorded motion is
   0.357 m; simulated motion 0.098 m; final positions 0.406 m apart. Because the
   arm never contacts the block, the block's final position is decided purely by
   settling and is therefore **independent of the actions**, so the goal rollout
   and the evaluation rollout land bit-identically and every distance is exactly
   zero.

   Open question being answered now: does the raw dataset store enough to
   restore the arm exactly (full qpos/qvel), or only the 14-dim end-effector
   state, which would need inverse kinematics and would not be exact?

The four cube evaluations that were submitted (26972104–07) were **cancelled**
— they would have burned machine time producing zeros.

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

## Seed 1 is fully evaluated — and the result is a null

Every arm has 100 scored goals (rope shuffled-depth has 90; a top-up job for
goals 90-99 is queued). Success rate:

| task | colour-only | real depth | zero depth | wrong-moment depth | no planning |
|---|---|---|---|---|---|
| rope | 0.44 | 0.50 | 0.46 | 0.13 | 0.02 |
| granular | 0.60 | 0.60 | 0.55 | 0.02 | — |
| cube | 0.99 | 0.97 | 0.98 | 0.13 | 0.00 |

Paired bootstrap, depth minus colour: rope +6.0 pp [-5, +17], granular
+0.0 pp [-10, +10], cube -2.0 pp [-5, 0]. Rope's upper limit sits above the
study's own 10-point threshold, so seed 1 cannot yet exclude the effect the
study exists to detect. It is "no evidence of a benefit", not "evidence of no
benefit".

Three measurements now constrain the interpretation:

1. The frozen encoder responds to depth as strongly as to RGB — relative
   feature displacement 0.162 vs 0.138 on rope, 0.180 vs 0.184 on granular
   (`scratch/depth_sensitivity.py`, job 26975609). It is not blind, and
   re-declaring the affine to a tabletop range moves the response under 10%,
   so a scale mismatch is not hiding the effect either.
2. Depth carries geometry RGB does not. A full-resolution 224^2 patchwise ridge
   predicting the depth residual from RGB scores test R2 of 0.35 to -1.32 on
   rope and 0.02 to 0.58 on granular. The earlier 32x32 pooled probe's R2 ~0.95
   was a smoothing artefact and must not be cited.
3. Deleting depth costs nothing; corrupting it is catastrophic (-33 to -84 pp).

Because each arm is *trained* under its own depth condition, a constant zero is
ignorable: that model simply learned an RGB-only solution and matched. Wrong-
moment depth cannot be ignored, because it is scene-plausible but wrong, so it
poisons the predictor. The finding is that the predictor does not exploit the
extra geometry even though it is present and the encoder sees it. The measure
itself is sound: it spans 0.00 to 0.13 to 0.50 to 0.99.

Camera geometry is not the explanation. All four PyFleX rope/granular cameras
sit at the same 45-degree elevation, differing only in azimuth
(`env/deformable_env/src/sim/sim_env/cameras.py`), so the original four-camera
probe was one viewpoint tested four times. OGBench-Cube renders from
`cube_env.py:539` 'front' at 20 degrees — already near-grazing — and shows the
identical null. (`scene_env.py`'s 38.8-degree 'front' belongs to the *scene*
task, not the cube.) A grazing-camera regeneration was proposed and cancelled.

## What is running

Fourteen jobs, all submitted 2026-08-26 on `mlgpu_medium` (free, uncontended):

- **Seed-2 evaluation, 12 runs**, `27163640`-`27163651`, `--time=23:55:00`.
  This needed a one-line launcher repair: `tools/plan_run.sbatch` searched only
  `matched-seeds-20260807` and `seed1-matched-20260806` for a checkpoint, so
  every seed-2 job would have exited 4. It now searches
  `matched-seeds-seed23` as well.
- **Rope shuffled-depth seed-1 top-up**, `27163655`, `goal_shard_start=90
  goal_shard_count=10`. Its original job timed out at 7:55 with 90 goals
  written; `aggregate_planning.py` merges shard files by global goal index.
- **Plan-time depth-ablation smoke**, `27163665`, 3 goals on rope.

Seed-1 timings, for sizing: rope 6:54-7:55 (one TIMEOUT at `mlgpu_short`),
granular 10:17-16:21. Everything now runs at 23:55 to remove that failure mode.

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
