# DINOcular compute budget

Updated: 2026-07-15T16:46:38+02:00

## P2 DINOv2 timing scope

The four authoritative Marvin cards are real A100 measurements for the RGB-only `dinov2_vits14` arm through the proven deterministic production step loop. Each job used one process, `env.num_workers=0`, global batch 32, deterministic algorithms, deterministic cuDNN, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, frozen pinned DINOv2 ViT-S/14, predictor depth 6, 16 heads, MLP width 2048, predictor LR `5e-5`, decoder off, seed 1, PushT history 3, and history 1 elsewhere. PushT and Wall used frame skip 5; Rope and Granular used the paper-correct frame skip 1. Each card excluded 20 warmup steps and measured the next 200 production-loop optimizer steps. Evaluation and profiling were absent, periodic checkpointing was disabled, and the final checkpoint was written only after the measured clock stopped.

These strict rates replace the provisional DONE-19 rates for P3 projections. Window counts come from both the runtime loader and an independent released-data calculation. Epoch rates and walls are projections from the measured fixed-step rates, not directly timed complete epochs. Only the DINOv2 rate columns are MEASURED; all target and matrix walls remain PROJECTED.

## MEASURED strict-path rates and PROJECTED epoch equivalents

| Environment | Marvin job | Train windows, VERIFIED FROM DATA | Measured wall, 200 steps, MEASURED | Steps/s, MEASURED | Samples/s, MEASURED | Epochs/h, PROJECTED | Epoch wall, PROJECTED | Peak process GPU memory, MEASURED MiB | Final finite loss, MEASURED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PushT | `26516573` | 1,981,721 | 409.8099 s | 0.488031 | 15.616996 | 0.028370 | 35:14:55.15 | 16,684 | 2.271771 |
| Wall | `26516574` | 70,848 | 134.0734 s | 1.491720 | 47.735036 | 2.425561 | 00:24:44.19 | 4,116 | 2.554715 |
| Rope | `26516575` | 17,100 at frame skip 1 | 71.8165 s | 2.784875 | 89.116001 | 18.761263 | 00:03:11.88 | 4,038 | 1.952680 |
| Granular | `26516576` | 17,100 at frame skip 1 | 70.2718 s | 2.846093 | 91.074978 | 19.173680 | 00:03:07.76 | 4,126 | 2.009485 |

All four jobs completed with exit `0:0` on `sgpu002`, one A100 80 GB per job. Their SLURM elapsed times were 485, 189, 118, and 119 seconds, totaling 0.2531 MEASURED A100-hours. PyTorch peak reserved memory within the measured window was 16,058, 3,490, 3,412, and 3,500 MiB. The table reports the independently sampled current-process peaks, which include warmup and the measured interval.

The independent calculation reproduced PushT as `sum_i(T_i - 4 * 5 + 1) = 1,981,721` over all 18,685 released training trajectories and Wall as `1,728 * (50 - 2 * 5 + 1) = 70,848`. Rope and Granular each reproduce `900 * (20 - 2 * 1 + 1) = 17,100`. The runtime loaders recorded the same four counts, so the DONE-19 Rope/Granular configuration-mismatch tags are removed.

## Upstream execution audit

The 100-epoch PushT projection is not upstream-faithful. Upstream code defines an epoch as a complete dataloader pass over every valid sliding window. The released PushT run metadata requests 100 epochs, effective batch 32, and per-GPU batch 1, but the released `model_latest.pth` contains `epoch = 2`. Its predictor optimizer counter is exactly 123,858, equal to `2 * ceil(1,981,721 / 32)`. The released model therefore executed two full sliding-window epochs, not 100.

The released Wall checkpoint separately contains `epoch = 65` and 36,010 optimizer steps at effective batch 128, exactly `65 * ceil(70,848 / 128)`. P3 retains its fixed global batch 32 and matches the release's 65 complete data exposures, producing `65 * ceil(70,848 / 32) = 143,910` steps. Rope and Granular have no released checkpoints. Their paper-faithful fallback remains 100 epochs at batch 32 and frame skip 1, or 53,500 steps each.

## Corrected P3 projections from MEASURED step rates

| Environment | Upstream-faithful P3 length | Steps per cell | One DINOv2 cell, PROJECTED A100-h | Nine cells across three arms, PROJECTED A100-h |
|---|---|---:|---:|---:|
| PushT | 2 complete epochs, released checkpoint exact | 123,858 | 70.498 | 634.478 |
| Wall | 65 complete data exposures, released checkpoint exact | 143,910 at P3 batch 32 | 26.798 | 241.181 |
| Rope | 100 epochs, frame skip 1, paper fallback | 53,500 | 5.336 | 48.027 |
| Granular | 100 epochs, frame skip 1, paper fallback | 53,500 | 5.222 | 46.994 |
| Total | 36 cells | | 107.853 across one cell per environment | 970.681 |

Projection formula:

```text
projected hours(env) = target optimizer steps(env) / measured DINOv2 steps/s(env) / 3600

DINOv2 12-cell arm = 3 seeds * sum_env(projected hours(env))
                    = 323.560 A100-h

36-cell P3 matrix  = 3 arms * DINOv2 12-cell arm
                    = 970.681 A100-h
```

The DINOv2 rows are PROJECTED from MEASURED DINOv2 step rates. Both DINOcular arms carry `[ASSUMPTION: PROJECTED-FROM-DINOV2]`; their rates and memory may differ because of encoder geometry and depth-input behavior. The corrected DINOv2 frame-skip cards do not measure either DINOcular arm. Replace those remaining assumptions only after arm-specific P2 cards pass. No DINOcular training rate or projected DINOcular row is labeled MEASURED.

The superseded 15,172.454 A100-hour figure was arithmetically correct for a hypothetical 100-full-pass, frame-skip-5 matrix. The later 503.152 A100-hour projection used non-strict DONE-19 rates. Neither superseded figure may be used for strict production scheduling.

## Twenty percent reserve gate

| Quantity | A100-h | Status |
|---|---:|---|
| P3 matrix projection | 970.681 | PROJECTED |
| Retry reserve at 20 percent | 194.136 | PROJECTED and required by plan |
| Required production plus reserve | 1,164.817 | PROJECTED monitoring baseline |
| Marvin hard GPU-hour allocation | No finite cap configured | PASS, fair-share usage monitoring applies |

At 2026-07-15T13:32:37+02:00, live `sacctmgr` reported association `ag_ifi_blum`, QoS `normal`, and blank `GrpTRESMins`, `GrpTRESRunMins`, `MaxTRESMins`, and `MaxTRESRunMins`; the `normal` QoS also has no GPU-minute cap. `sshare -A ag_ifi_blum -l` reports decaying fair-share usage rather than a remaining-hour balance. The official Marvin wiki page `https://wiki.hpc.uni-bonn.de/research-groups`, fetched from the Marvin login node as 21,856 bytes with SHA-256 `a32ca7c4880f7ed485cd8dd4cc71ec239458651e2f178c0bc6936a848ec70eba`, states that using up a contingent does not stop group jobs, only reduces their priority, and that the contingent regenerates continuously. This proves there is no finite hard GPU-hour allocation gate for the current Marvin association. It does not promise immediate scheduling or unlimited concurrency.

## Marvin A100-hour usage ledger

Root jobs are counted once as `ElapsedRaw * allocated A100 count / 3600`; batch, extern, and step records are excluded. This ledger is updated per phase from `sacct` and is an operational usage record, not a quota balance.

| Phase | Terminal root jobs through 2026-07-15T16:46:38+02:00 | Elapsed A100-seconds | Consumed A100-h, MEASURED |
|---|---|---:|---:|
| P0/P1 A100 works-proof | `26511327`, `26511347`, `26511353`, `26511359` | 136 | 0.0378 |
| P0 DA3 canaries, retries, and failed first production | `26511360`, `26511365`, `26511375`, `26511379`, `26511387`, `26511398`, `26511399`, `26511413` | 1,809 | 0.5025 |
| P2 provisional DINOv2 timing | `26515714` to `26515717` | 562 | 0.1561 |
| P2 strict production-path DINOv2 timing | `26516573` to `26516576` | 911 | 0.2531 |
| P0 MapAnything canary and validation attempts | `26515725`, `26516234`, `26516242`, `26516447`, `26516457`, `26516529`, `26517855`; stale `26516530` canceled before allocation | 1,055 | 0.2931 |
| P3 resume engineering and proofs | `26516445` canceled before allocation; `26516460`, `26516461`, `26516519`, `26516525`, `26516526`, `26516528` allocated | 245 | 0.0681 |
| Total measured A100 use | all rows above | 4,718 | 1.3106 |

Pending jobs consume zero until allocation. Resume attempt `26516445` was canceled with `ElapsedRaw=0` and no allocated TRES, so it is recorded but adds zero. Add every completed or failed production shard to the applicable phase row and retain the 20 percent retry projection as a planning margin.

## MapAnything full PushT cache projection

The exact released PushT inventory contains 18,706 trajectories and 2,339,250 frames. The invalidated batch-8 canary measured 4.471696604141974 committed frames/s. At that producer rate, serial generation would project to 145.312 A100-h and the 20 percent reserve would project to 174.375 A100-h. These figures are retained as superseded evidence only: independent-batch equivalence failed at max absolute error `1.364849328994751 > 0.001`, so they cannot budget the literal-singleton retry or its full cache.

The eight-shard batch-8 graph `26516463` to `26516471` was canceled before allocation after that hard-gate failure and consumed zero A100-hours. Literal-singleton generation `26516529` completed 1,090 frames in 390 SLURM seconds and recorded 4.100143591125776 committed frames/s in immutable manifest `080ecbe4-23da-426f-b884-ce0ce6ae4e18`. Validator `26517855` completed in 77 A100-seconds and passed every applicable hard gate; its temporal failure remains characterization-only under Amendment 4. The generation plus validator consumed 467 A100-seconds, both included above. Stale validator `26516530` was canceled before allocation and consumed zero.

The deterministic eight-shard literal-singleton plan covers all 18,706 trajectories and 2,339,250 frames exactly once. Plan SHA-256 is `4569ae4eb0cd004d293f88f308e694deb5ffc6dcdd45a7dacea87352ae48a23f`; the materialized plan file SHA-256 is `966f86f79d8e63f0fbba70b29117c2a365e6d6efcb5c013f10995579382482df`. Shards contain 292,391 to 292,440 frames. Using measured startup of 124.15565192420036 seconds once per shard, the largest shard projects to 71,448.486983 seconds including startup and 85,738.184380 seconds with 20 percent reserve, or 23.816162 hours. It does not fit one-hour `sgpu_devel` or eight-hour `sgpu_short`; it fits 24-hour `sgpu_medium` with 661.815620 seconds remaining beyond reserve. The eight generation shards project to 158.756123 A100-hours before reserve and 190.507348 A100-hours with reserve. Merge and full validation consumption is additional and will be measured from terminal accounting rather than invented.

Production shards `26518202` to `26518209` are pending on `sgpu_medium`; all-gate merge-validator `26518210` is `afterok` all eight. Pending roots consume zero until allocation. Each shard uses batch size 1, immutable calibration SHA-256 `4d65403a...`, distinct output roots, and unchanged producer, model, source, wrapper, and SIF provenance. The accepted canary and eventual normalized full cache remain best-evidence P2a pilot candidates only, not student-native activation-ready.

Step-level sampler and checkpoint resume is implemented and proven. Under the strict rate, a PushT cell projects to 70.498 hours total and one complete PushT epoch projects to 35.249 hours, both above the authorized `sgpu_short` maximum of 8 hours. P3 must use the proven deterministic sub-epoch chunks and resume the exact sampler cursor, optimizer, scheduler, and RNG state. No `_long` partition is authorized for these training cells.

This budget covers the 36-cell P3 training matrix only. The two-cell P2a producer pilot uses the same 123,858-step PushT target but remains separately budgeted. Cache production, P4, P5, P6, and the D-6 control also remain separate as required by `MASTER_PLAN.md`.

## Evidence

- Audit report: `reports/14_budget_audit.md`
- Resume report: `reports/15_resume_implementation.md`
- Strict timing report: `reports/16_strict_p2_timing.md`
- Upstream code commit: `0a9492fa12044b852ae9e001cc74604b79c8bb0c`
- Strict resume base commit: `12374e46f27cb9074afe5e3973b78c5561f05657`
- Strict timing commit: `33970180f64d45672aa26d40e2ccf95cf5a95550`
- Production timing hook SHA-256: `a8c8379a7210080b9945d0e72b54200b334c30ca83b376379e32284362dc4ec3`
- Strict SLURM wrapper SHA-256: `dc4bd3b37688b0925732ed17d088db33211c34747fb60594c91f1b3c88b518a0`
- Independent collector SHA-256: `250347d5c175313f87b1a0567252d4eca5a3333de71dbb9275828679faf1eb40`
- Apptainer SIF SHA-256: `6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae`
- DINOv2 weight SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- Released checkpoint archive SHA-256: `425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482`
- Released PushT checkpoint SHA-256: `e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5`
- Released Wall checkpoint SHA-256: `8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b`
- Marvin result root: `/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/results/p2/strict_dinov2_vits14`
- Timing summary SHA-256: `771ac83fe23773f390ca3b653f06825675ebe2d37267e837fd7f53e7a2b19167`
- One-page plan SHA-256: `bf0ce55463274ba52059b108781eb23e6ecdaf500d597229fb16f4b787131551`
- Per-environment records: `timing.26516573.json`, `timing.26516574.json`, `timing.26516575.json`, and `timing.26516576.json`
