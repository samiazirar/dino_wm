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

## What is queued

- Eight rope/granular evaluations `26950256`–`26950263`, `sgpu_medium`,
  `--time=23:55:00`, estimated start 15–16 August. Kept as a fallback. They
  should be replaced by sharded jobs on the free `mlgpu_*` (A40) partitions
  once sharding is committed.
- `mlgpu_*` is free and fast; `sgpu_*` is congested. `mlgpu_devel` accepts
  short probes.

## Exact next action

1. Confirm the sharding commit and its two probe jobs (different goals, global
   indices 0 and 1).
2. Submit the eight rope/granular evaluations as shards on `mlgpu_*`, sized
   from the measured per-goal minutes, with all four arms receiving identical
   goals per shard.
3. Settle cube arm restorability. If the arm is exactly restorable from disk,
   extend `set_state` and re-run the cube smoke; the models need no retraining
   because state is used only for env restore and scoring. If it is not, the
   cube contributes training only and that must be stated plainly.
4. Merge shard outcome files by global goal index and compute the pooled
   bootstrap comparison.
5. Queue seeds 2 and 3 only after step 4 shows the measure separates systems.

## Repository state

- `code/dino_wm` on marvin at `4f865ff`, with `plan.py` carrying the
  uncommitted goal-rejection plus sharding work (being committed).
- Untracked helper scripts remain in `code/dino_wm` and `scratch/`.

## Notes

- DeepSeek workers run through `claix-deepseek` against the self-hosted
  endpoint; the prepaid API balance is negative.
- The cube env runs behind a multiprocess vector env (`env/venv.py`). Printing
  inside it kills the worker with `EOFError`; diagnose the cube with standalone
  single-process scripts instead.
- Prediction error is not reported. Planning success is the single measure.
