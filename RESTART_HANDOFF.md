# DinocularWorldModel restart handoff

Status: CAN RESTART
Last updated: 2026-08-11

## Goal

Does depth supplied at the moment of use improve world-model prediction and
robot planning? Judged by planning success across four visual systems
(dino_pinned, dinocular, dinocular_zerodepth, dinocular_shuffleddepth) on three
tasks (rope, granular, ogbench_cube), three seeds — 36 runs.

## Where the real work lives

Everything current is on Marvin, not in the local checkout.

- Cluster project root:
  `/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm`
- Cluster code (authoritative): `<root>/code/dino_wm`, HEAD `338a0b4`
- Local `~/dinocular-wm/dino_wm` is **stale and divergent** (uncommitted edits on
  top of `ce55391`, 2026-08-06). Do not treat it as the source of truth; pull
  from the cluster checkout before touching code.
- Docs repo: `~/dinocular-wm-worktrees/project-docs`, branch
  `task/human-plan-current-update`, pushed.

## Current state — measured

**Training: first seed complete, all twelve runs at step 53,500 / 100 epochs.**

- rope + granular, dinocular / zerodepth / shuffleddepth:
  `<root>/outputs/seed1-matched-20260806/{rope,granular}-<arm>-s1` (2026-08-07)
- rope + granular, dino_pinned:
  `<root>/outputs/campaign-seed1/rgb-dino-seed1-20260728a/{rope,granular}/run`
- ogbench_cube, all four arms:
  `<root>/outputs/matched-seeds-20260807/ogbench_cube-<arm>-s1` (2026-08-10)

Seeds 2 and 3 (24 runs) not started.

**Depth at planning time: implemented and verified.** Commits `4b5dbed` (keep the
renderer's 5th channel as `obs["depth"]`, `env/deformable_env/FlexEnvWrapper.py`
prepare/step_multiple), `d20ddb5` (`plan.py` `depth_lo_hi_from_contract`,
`raw_metrics_to_wire_depth`, `PlanDepthPreprocessor.transform_obs`,
`PlanWorkspace._prepare_depth`), `d57bb42` (same-episode substitution for the
shuffled arm). Wire conversion `clip((d - lo)/(hi - lo), 0, 1)` at `plan.py:108`
is identical to the training producer `tools/precompute_depth.py:691-697`; lo/hi
come from the same native depth contract binding. Rope live render spans
9.51–24.06 m against contract 9.901+14.163 = 24.064.

**Planning outputs that exist** (`per_goal_outcomes.json`, tiny n, all success
1.0; thresholds rope 1.132036, granular 1.770630, ordered metric):

| run | n | ordered distances |
|---|---|---|
| `seed1-matched-20260806/plan-rope-dinocular-s1` | 2 | 0.485, 0.693 |
| `seed1-matched-20260806/plan-rope-dinocular_shuffleddepth-s1` | 1 | 0.487 |
| `seed1-matched-20260806/plan-granular-dinocular-s1` | 1 | 1.083 |
| `campaign-seed1/rgb-dino-seed1-20260728a/rope/plan-rope-dino_pinned-s1` | 1 | 0.480 |

Ceiling risk: everything passes on these goals. Check the pass rate on the first
full 100-goal run before drawing any comparison.

**2026-08-08 failure and repair.** All eight full evaluations hit the 8 h limit
and saved nothing, because scores were written only after all 100 goals.
`338a0b4` scores goals in chunks of 10 and persists after each chunk, with a
per-chunk timing line. Proved by a deliberately time-limited run
(`plan-chunk-smoke` 26948623) whose finished goals survived on disk. Why >8 h was
needed is still unexplained; the new chunk timings will show it.

## What is running

Eight full evaluations queued 2026-08-10, `sgpu_medium`, `--time=23:55:00`:
`26950256`–`26950263` (rope/granular × 4 arms, seed 1).
Scheduler estimate: **start 2026-08-15**. Shorter walltime does not move them.

Two 1-hour probes on `mlgpu_medium` (A40), submitted 2026-08-11, estimated start
~22:28 the same day:

- `26967707` `plan-a40-probe` — rope/dinocular, `n_evals=2`. Tests whether the
  planning job runs on A40 at all.
- `26967708` `plan-cube-smoke` — ogbench_cube/dino_pinned, `n_evals=2`. Cube
  planning has never been run.

`mlgpu_devel` and `sgpu_devel` are drained; do not submit probes there.

## Exact next action

1. Read both probe logs when they finish:
   `<root>/logs/plan_plan-a40-probe.26967707.{out,err}` and
   `<root>/logs/plan_plan-cube-smoke.26967708.{out,err}`.
2. If the A40 probe succeeds, resubmit the eight evaluations on `mlgpu_medium`
   with `--time=12:00:00` and cancel the `sgpu_medium` duplicates — that moves
   first results from 15 August to the next day.
3. If the cube smoke succeeds, queue four cube evaluations the same way.
4. When the first full evaluation returns, check the pass rate before anything
   else. Near-total success means the goals are too easy and must be hardened
   before the comparison means anything.
5. Queue seeds 2 and 3 only after step 4 shows the measure separates systems.

## Repository state

- `code/dino_wm` on Marvin at `338a0b4`, with untracked helper scripts
  (`aggregate_planning.py`, `done_check_cube.py`, `planpath_check_cube.py`,
  `step_check_cube.py`) not yet committed.
- `project-docs` at `144f712`, pushed to
  `origin/task/human-plan-current-update`.
- Local `dino_wm` working tree has uncommitted edits that predate the cluster
  commits; reconcile against the cluster before using it.

## Notes

- The prepaid DeepSeek API balance is negative and the CLAIX self-hosted
  endpoint is unavailable while RWTH is in system maintenance. Free DeepSeek V4
  Flash workers run through `opencode --model opencode/deepseek-v4-flash-free`.
- Prediction error is not reported and will not be; planning success is the
  single measure.
