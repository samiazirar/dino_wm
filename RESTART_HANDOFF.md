# DinocularWorldModel restart handoff

Status: CAN RESTART
Last updated: 2026-08-11 (late)

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

Seeds 2 and 3 (24 runs) not started. Hold them until the first seed's
evaluation is shown to separate systems.

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

## What is running

Eight seed-1 evaluations, all started 2026-08-12, `n_evals=100`:

- rope × 4 arms: `26973067`–`26973070`, `mlgpu_short`, `--time=07:55:00`.
  Setup (100 goal rollouts) takes ~110 min, then ~30 min per 10-goal chunk.
- granular × 4 arms: `26974609`–`26974612`, `mlgpu_medium`,
  `--time=23:55:00`. Moved off the 7:55 limit deliberately: granular carries
  12,774 particles against rope's 1,965 and had not finished goal construction
  in 2.5 h, so at `mlgpu_short` it would have died having persisted nothing.

The eight old `sgpu_medium` jobs `26950256`–`26950263` were cancelled; they
predate every repair above. `mlgpu_*` is free and starts immediately;
`sgpu_*` is congested.

**First real discriminating scores** (rope, first 10 goals): colour-only
`[T,F,T,T,T,F,F,F,F,T]` = 5/10, real depth `[T,F,T,T,F,T,F,F,F,T]` = 5/10 —
agreeing on 8 of 10 goals individually. Distances span 0.275 to 2.271 against
the 1.132 cutoff. Before the goal repair every goal passed.

## The analysis would have reported nothing

`aggregate_planning.py` parsed run directories as `<task>-<arm>-s<seed>`, but
plan.py writes `plan-<task>-<arm>-s<seed>`. Every real run was skipped and it
printed "No run directories matched" — it would have produced nothing at the
end of the whole evaluation. Fixed in `256cca8`.

The same commit makes the bootstrap **paired** over goal indices, which is both
the literal reading of the settled rule and much tighter now that goal identity
across arms is verified: on the first ten rope goals the 95% range narrows from
±40 to ±30 points. It falls back to independent resampling only when two arms
differ in goal count, which happens only while runs are in flight.

## Exact next action

1. Watch the eight runs. Rope should finish inside its 7:55; if a run is cut
   off, chunked persistence keeps every completed goal and the remainder can be
   topped up with `goal_shard_start` / `goal_shard_count`. Note sharding does
   NOT reduce setup cost — each shard re-rolls all 100 goals — so prefer one
   long job per run and use shards only to fill gaps.
2. Run `python3 aggregate_planning.py <root>/outputs` for the pooled paired
   comparison.
3. Repair the cube (below). It is the only remaining task-level gap.
4. Queue seeds 2 and 3 only after step 2 shows the measure separates systems.

## Cube repair — recovered, not yet applied

The missing source `cube-single-play-v0.npz` was re-downloaded and sits at
`<root>/code/.ogbench_data/cube-single-play-v0.npz`. It holds the full joint
state (21-dim qpos, 20-dim qvel) the stored episodes lack. Re-running is
deterministic for physics (only colour rendering differs slightly, which does
not matter because images are never rewritten) and costs ~0.07 s per episode,
about two minutes for all 1000.

Remaining work: append per-frame qpos/qvel to each episode h5 using the
generator's own alignment (`tools/generate_ogbench_dataset_expert.py:140`
`_best_window`, `OBJECT_QPOS = slice(15, 18)`, frame `t` ↔ npz index
`start + t`); make `set_state` write the full state and call `mj_forward`;
prove one episode's replay follows the recorded block path (final distance was
0.406 m off); then rerun the 2-goal smoke and check the distances are no longer
exactly zero. Nothing needs retraining — state is used only for env restore and
scoring. Do NOT rewrite images, depth or actions.

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
