# GT-depth secondary track status

## 2026-07-16 Phase 0

- Scope: exact PyFlex replay feasibility for released Rope and Granular trajectories.
- Completion gate: sampled replay frames must show pixel-level RGB agreement with released RGB for both environments. Approximate alignment is forbidden.
- Main-campaign gate: `STATUS.md` does not show P3 dispatched. Phase 2 training is closed.
- Scheduler observation: main `dinocular-depth` and `dinocular-depth-ma` jobs occupy or depend on sgpu capacity. This track will not use A100 capacity during Phase 0.
- Forensic correction: the first accepted-action-only analysis was insufficient because the deterministic original retry loop can regenerate rejected candidate actions and their physical effects. Its provisional infeasibility verdict is not accepted.
- Correct replay probe: job `26542970`, `gtrender-full-loop`, one A40 on `mlgpu_devel`. It reruns the original seeded acceptance loop for Rope and Granular episode 0 and requires exact RGB agreement at frames 0, 1, 7, and 19.
- Topology update: orchestrator pane `w3:p7` and reusable Phase 0 worker `w3:p8` now live in tab `gt-depth-orchestrator` (`w3:t5`). Future new worker tasks must use one new Herdr tab per task.
- Terminal evidence 2026-07-16: job `26542970` (`gtrender-full-loop`) was `FAILED`, exit `255:0`, elapsed `00:01:25`, on `mlgpu016` with one A40. A40 telemetry showed 343 volatile uncorrectable ECC errors; Flex CUDA initialization failed before replay output. Script hashes matched expected values; log SHA-256 values were `7c65b070ddc17c8bcb5bbdb5838bc319b9f473dacfad7165b3f359bd36b1cae4` (out) and `0ca22204fb46875b1ce3b9f207e0728bb7521fcbf55258e6455ba375e76168b8` (err). Both metrics JSON files are missing.
- State: PROBE_INFRA_FAILURE. Budget-emergency pause. No pending job remains and no replacement will be submitted.
