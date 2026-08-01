# DinocularWorldModel restart handoff

Date: 2026-07-23

## Current legacy inspection and live-state correction — 2026-08-01

This section supersedes older execution, pane-topology, scheduler, and watcher
assertions below. Research remains paused; this inspection did not resume
coding, testing, experiments, evaluation, paper work, or campaign control.

- The fixed question is unchanged: whether useful depth input improves
  world-model prediction and robot planning across the locked DinocularWorldModel
  comparison.
- Every pane in legacy Herdr workspaces `w2T` (`Claude RC`) and `w2V`
  (`Command`) was read. The full disposition table and cross-project pointers
  are in `OLD_HISTORY.md`; the only substantive claims were unrelated
  DFormerv2/JUPITER scaling and Kimi decode-speed claims. Neither changes
  Dinocular evidence or the paused scientific state.
- Superseded Human session
  `019f8f33-ec04-7962-82bd-3a1a80c90a75` and superseded Operations session
  `019fb1c0-ad48-7a93-a965-2d95c0c42b47` are archived in `OLD_HISTORY.md`.
  Neither appears in the current Herdr pane inventory. The observed current
  Dinocular leadership was Human `w3:p8K`, Operations `w3:p8M`, and Selene
  `w3:p8N`.
- Read-only Marvin snapshot at `2026-08-01T15:50:29+02:00` found exactly two
  continuing seed-one PushT jobs: `26792098`
  (`countable-pusht-dinocular-s1`, `sgpu021`) and `26792308`
  (`countable-pusht-dinocular_zerodepth-s1`, `sgpu023`). Their exact sbatch,
  stdout, and stderr paths are recorded in `OLD_HISTORY.md`. They remain
  untouched.
- The detached controller remains PID `874135` with child `874137`, running
  `/tmp/dinocular_resume_controller.py` from the deleted
  `rope-granular-admission-hold` worktree. Its state file was updated at
  `2026-08-01T13:56:44Z` and identifies the two PushT arms as active seed-one
  resumes. It remains untouched.
- Two older watcher loops are still present in the local process table (PIDs
  `5358` and `1182620`) but target absent panes and stale/empty inputs. Their
  ownership is unresolved and no watcher action was taken.

No live job, controller, watcher, repository, result, or other project state
was changed by this recovery inspection.

## Current execution update — 2026-07-29

This section supersedes the July 23 stop-state and execution-state assertions
below. The remainder of this handoff is preserved as useful restart history.

- Rope DINOv2 seed one was accepted without retraining from its deterministic
  step-53,500 checkpoint. Acceptance job `48183` completed successfully, and
  the checkpoint, training ledger, and validation ledger were hash-bound. The
  accepted artifacts were copied into the existing Marvin lineage and checked
  byte-for-byte. Its fixed held-out prediction evaluation is complete with 100
  episode records.
- The deterministic-step planning repair was integrated as campaign commits
  `8ddca47` and `09b8375`. The real builder produced 36 planning cards, 36
  wrappers, 36 lineages, and 1,080 expected outcomes with no missing input and
  without changing targets, planner settings, checkpoints, prediction outputs,
  or scientific settings.
- A later Hydra goal override repair changed only `goal_file_path=...` to
  `+goal_file_path=...` and was integrated as campaign commit `37e4503`. The
  regenerated planning set retained the same 36 cards, 36 wrappers, 36
  lineages, and 1,080 expected outcomes.
- The corrected Granular and Rope DINOv2 seed-one relaunches, jobs `26717065`
  and `26717066`, both failed before planning. Hydra's output-directory
  interpolation calls `replace_slash(model_name)`, but deterministic-step
  launches do not set `model_name`, causing
  `AttributeError: 'NoneType' object has no attribute 'replace'`. Neither
  expected `planning_results.jsonl` exists. The watcher record is
  `/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/campaign-planning/planning36-20260728c/watcher/pair-26717065-26717066.26717094.out`.
  The smallest remaining repair is a runtime-safe Hydra output directory or an
  operational `model_name` that leaves planning semantics unchanged.
- Weights & Biases project `rlp_uni_bonn/dinocular-wm-campaign` is confirmed at
  <https://wandb.ai/rlp_uni_bonn/dinocular-wm-campaign>. The visible imported
  lineage `granular/dinocular/seed3` is at
  <https://wandb.ai/rlp_uni_bonn/dinocular-wm-campaign/runs/dc-176b170c0f2391492014>
  with 1,812 imported records; a repeat import added zero records. Imported and
  live provenance are distinct. Telemetry is fail-open, runs outside training,
  and cannot stop the controller or GPU jobs.
- Exactly one controller process is active from
  `/tmp/dinocular_resume_controller.py`; healthy training jobs were not
  restarted or modified.

## Stop state

The project is paused for restart. Do not resume research, coding, testing, scheduler work, experiments, or paper editing until the user explicitly says to continue.

All worker panes have been closed. No worker or shell is waiting. No recurring task is scheduled. The only preserved Herdr contexts are:

- Primary orchestrator: pane `w3:p10`, tab `w3:t1N`, label `claudex-orchestrator-high`.
- Old reserved advisor: pane `w3:p1P`, tab `w3:t1F`, session `019f75a1-7f32-71d1-ab61-cb1a87e6e773`.

The old advisor is historical context only. Use it only when old code or an old decision is unclear.

## Actual project goal

The goal is to obtain real, reproducible empirical results for the locked DinocularWorldModel study and finish the intended empirical paper. A protocol paper, execution plan, or set of passing local tests is not the final deliverable.

The locked study is:

- Arms: `dino_pinned`, `dinocular`, and `dinocular_zerodepth`.
- Environments: `pusht`, `wall`, `rope`, and `granular`.
- Paired seeds: `1`, `2`, and `3`.
- Total: `3 arms x 4 environments x 3 seeds = 36 cells`.
- Global batch size: `32`.
- Training targets:
  - PushT: `123858` optimizer steps.
  - Wall: `143910` optimizer steps.
  - Rope: `53500` optimizer steps.
  - Granular: `53500` optimizer steps.

PushT has a special, explicitly limited recovered contract:

```text
informative value = float32(wire_f16) * 1.5746406149864196
zero-depth value = exact float32 zero after validating the same payload
zero-depth audit mask = exact zero
mandatory label = [ASSUMPTION: RECOVERED-CONTRACT]
```

The project must not claim that this recovers the original producer, is checkpoint-native depth, is physical or metric depth, is a neutral-depth or RGB-only equivalence test, or proves MapAnything better than Depth Anything 3. Wall, Rope, and Granular retain their previously defined semantics. The scale, thresholds, seeds, metrics, horizons, and matrix must not be changed after seeing outcomes.

## Objectively testable finish condition

The project is finished only when all of the following are true:

1. All 36 locked cells have completed their exact target steps under accepted immutable source, runtime, input, and run-card identities.
2. Every interrupted training lineage has proven deterministic resume continuity, including model, optimizer, scheduler, random-number-generator, sampler, data-order, step, and checkpoint identities.
3. Every cell has independently accepted evaluation evidence for the predeclared open-loop and planning procedures that apply to it.
4. A collector produces exactly 36 unique accepted rows with complete arm, environment, and paired-seed coverage.
5. No failed, altered, fallback, incomplete, or unaudited run is included.
6. PushT results preserve the recovered-contract label and all claim restrictions.
7. Every number inserted into the empirical manuscript is traceable to independently accepted evidence.
8. The rebuilt empirical manuscript passes a fresh independent scientific and mechanical review.

No part of that finish condition has yet been met at the result level. There are no accepted training or evaluation outcomes.

## What was attempted

### Infrastructure and inputs

The project prepared and audited substantial infrastructure:

- Accepted source base:
  - commit `158c74704bd18343dfbccb2fb25f4b6726c33f93`
  - parent `27b73d7ef750628b87aa3049b2bfb6a988cf110a`
  - tree `ac2cec54d090b72ef3be29d4caf5b7c390b42d0e`
- Historical immutable Git bundle:
  - SHA-256 `dc2bae175c0ec102a353098c06005722ef324190f9f908f7adff7d35055de358`
  - size `6126703` bytes
  - ref `refs/heads/accepted-158c7470`
  - receipt SHA-256 `f6951419a8d49d1052e5f5d49d46acadd3c8053e241eb2b8b3404d5081a49b5c`
- Runtime image:
  - `/home/data/sif_cache/dinocular-wm-p0-20260714.sif`
  - SHA-256 `6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae`
  - size `8149331968` bytes
- The accepted source snapshot previously passed `172 passed, 3 skipped` inside that image. The definitive transcript is `/home/data/sif_cache/dinocular-wm-p0-20260714.pytest-final.log`, SHA-256 `401c73b2a9555819eafb605512b6f877474a3a44243c9e05d585fe0c157981cd`.

The historical accepted snapshot predates the current dirty empirical implementation. It is useful provenance, but it is not an accepted release of the current candidate.

### Data and model assets

Known pinned model identities include:

- DINOv2 repository commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- DINOv2 weights SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- DINOcular student checkpoint SHA-256 `decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc`.

The accepted PushT MapAnything cache is:

```text
/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/data/depth_cache_mapanything_singleton/pusht.lmdb
```

Its important identities are:

- manifest ID `b87fe658-4731-4e8e-8c88-38f4fac6344c`
- manifest SHA-256 `9f35d303a5c604d5870ebdc6aedcefe9860bd0ab2763648c75b11e3b4f50b691`
- validation SHA-256 `ed3d63388a81581d20d560266a0d5e210b6672144fcb2cbb6a2e7ffef5736df9`
- `data.mdb` SHA-256 `68fe99e9f566eadcaa61e092b042962060fdc82c39ee8e71dcd7b9ef05247adb`
- source-index SHA-256 `ed6eecd62e455ffd35551074787f36b6c08039776b58a9dc6f22d17266ead6bd`
- producer identity `6996aa719531feb09f8dc858e66c2b703dc14393533f2d6dc073fd98937fe4e5`
- frozen calibration `lo=0.0`, `hi=1.5746406149864196`

Historical Rope and Granular caches passed their independent cache checks and remain preserved and unconsumed. Their accepted records are in `STATUS.md`, `reports/23_granular_acceptance.md`, `reports/27_rope_direct_consumption_decision.md`, and the current governance reports. They must not be copied, promoted, aliased, or consumed without a separately reviewed exact-path and exact-hash consumption decision.

### Empirical implementation

The dirty implementation candidate added a separate lossy empirical-cache contract, immutable runtime bindings, exact receipt classification, host-path checks, PushT proxy and zero-depth handling, training resume evidence, evaluation bindings, and result-collection checks.

The main implementation and audit sequence was:

1. `reports/71_empirical_results_repair.md` claimed closure.
2. `reports/74_independent_empirical_results_repair_audit.md` rejected it because many authoritative paths still accepted aliases, the full-route test manufactured important identities, and a fresh-process test failed.
3. `reports/76_empirical_audit_closure.md` repaired many consumer paths, replaced fake hash proofs with real temporary bytes, and made the adjacent tests green.
4. `reports/78_independent_empirical_audit_closure.md` rejected that closure because the materializer still opened the producer decision, held-out manifest and metadata, and fixed evaluation manifest and metadata before lexical no-alias validation. It also found that the strongest route test entered after the vulnerable part of `make_training()`.
5. A final worker began task 68 to close those exact findings. The user ordered the restart before the worker finished.

The last worker was stopped and its pane was closed. It did not produce the expected file `reports/80_materializer_alias_closure.md`. Its visible partial work included:

- adding parameterized alias-rejection tests for fixed evaluation manifest and metadata inputs;
- beginning corresponding materializer hardening;
- beginning conversion of the full-route fixture to use a mixed native and empirical index and the complete `make_training()` route;
- adding the DINOcular student artifact to the temporary materializer fixture.

The worker had not finished routing the test through `make_training()`, had not run final verification, and had not written a report. These edits are unverified and must not be treated as closing report 78.

Before this unfinished final edit, independently reproduced evidence was:

- direct empirical suite: `63 passed`;
- adjacent suites in `.test-venv`: `129 passed`, zero skips;
- dedicated resume suites: `11 passed`;
- meaningful real-byte downstream route coverage through materialization helper, live verification, chain verification, cache reader, real encoder checkpoint, and receipt classification.

Those prior test counts do not validate the current partially edited tree. The current tree has not been tested after the final interrupted changes.

### Execution design

A long series of reports tried to define a secure execution route. The first approach became an over-engineered self-referential capsule design. Reports 41 through 59 repeatedly proposed and rejected variants involving circular hashes, future-dependent evidence, publication ordering, and incomplete bootstrapping. That work was stopped.

After the user reopened the project for actual results, the project switched to an ordinary hash-bound route:

- `reports/72_bounded_results_execution_route.md`: first conventional route.
- `reports/75_independent_bounded_execution_route_audit.md`: independent rejection of six design defects.
- `reports/77_corrected_bounded_results_execution_route.md`: corrected card hashes, external launch decisions, evaluation templates, receipt order, network denial, and training segments.
- `reports/79_independent_corrected_execution_route_audit.md`: independent rejection of three remaining defects.
- `reports/81_final_bounded_results_execution_route.md`: added a finite initial-wave versus expansion-wave rule, fixed scheduler-finalization policy in each card, and supplied a complete graph for the CPU probe, training, evaluation, collection, and paper eligibility.

Report 81 is SHA-256 `c124383e6def28d8750dffe3a3fc990d044bac45b4a817e0ed00eef834368b2a`. It self-reports readiness, but it has not received a fresh independent audit. It therefore supplies no current launch authority.

The execution designer's restart note is `reports/82_worker_restart_note_execution_route.md`, SHA-256 `c7e097c2d4dab50d2fe26232780f9e6aec255d872544907586cf2a0f57e7607e`. It explicitly states that report 81 remains unaudited.

### Paper work

Two paper artifacts must be distinguished:

1. Results-oriented draft:
   - `reports/ECCV_SUBMISSION_DRAFT.pdf`
   - SHA-256 `0ad346391a67532421548868c63b08a7f8564666c19bf9531ab23e152a1d4af5`
   - frozen historical draft containing unresolved result placeholders.
2. Protocol-only candidate created after the project goal was misread:
   - `reports/ECCV_SUBMISSION_FINAL.pdf`
   - SHA-256 `a7341ac6d15ab8e08888286a303734d6dade0ec6466dccc33011c8a685f0660a`
   - source `paper_final/main.tex`
   - source SHA-256 `8378cb146db4b79e27195bb3a2df24a5c1202d7a0ab7c07c28ccf7b58f29819e`

`reports/69_independent_final_paper_audit.md` passed the protocol-only artifact for the limited claims it made. The user explicitly rejected that artifact as the desired final deliverable. Preserve it as provenance only. It is not project completion.

No current empirical number has been added to an accepted final paper. No external submission was made.

## Concrete achievements

The following are evidence, not hopes:

- The locked scientific matrix, target steps, paired seeds, deterministic settings, PushT recovered-contract semantics, and prohibited claims are documented.
- Important source, model, container, cache, and historical bundle identities are pinned.
- The exact SIF exists locally and has a verified hash.
- PushT MapAnything cache bytes and validation evidence are accepted.
- Historical Rope and Granular cache evidence was independently checked and preserved.
- The code candidate contains substantial real implementation for empirical cache consumption, frozen encoders, deterministic resume, evaluation bindings, and collection.
- Earlier tests reached `63` direct passes, `129` adjacent passes with no skip, and `11` dedicated resume passes before the last interrupted edit.
- The earlier fake-hash route test was replaced with real temporary LMDB, source, manifest, checkpoint, and canonical-object bytes for the stages it covered.
- Consumer-side alias checks were substantially improved.
- The rejected recursive capsule direction was abandoned.
- Report 81 defines a much simpler conventional execution route with closed cards, external digests, later decisions, explicit network denial, finite receipts, and a first-four then maximum-twelve launch policy.
- All finished and idle worker panes were captured and closed.

## What has not been achieved

The following are facts:

- No independent implementation PASS exists for the current tree.
- The final materializer alias repair is incomplete and unverified.
- No current clean source release exists for the dirty empirical candidate.
- Report 81 has not been independently audited.
- No accepted supervisor or execution implementation exists for report 81.
- No CPU/no-GPU probe has been implemented and accepted for this current route.
- No geometry, deterministic-resume, timing, or memory canary has run for the current empirical candidate.
- No one-A100 training cell has launched.
- No one of the 36 cells has trained.
- No accepted evaluation exists.
- No accepted 36-row collection exists.
- No empirical result exists for manuscript inclusion.
- No empirical paper has been completed or submitted.

## Where the last several days went

### Directly useful work

- Preparing and hashing source, runtime, model, data, and depth-cache artifacts.
- Building native depth extraction support and testing the accepted historical source snapshot.
- Reconstructing the PushT empirical contract without claiming unavailable original provenance.
- Implementing real cache consumption, encoder, deterministic resume, evaluation, and collection paths.
- Replacing fake identity tests with real temporary byte-valid fixtures.
- Running direct, adjacent, resume, and focused test suites.
- Independent audits that found real release-blocking alias and test-coverage defects.
- Replacing the failed capsule concept with a conventional execution design.

### Detours that delayed results

- The original plan made the PushT route depend on a two-producer comparison. The Depth Anything 3 PushT route failed its provenance contract and missed its deadline, so much effort went into failed shards, immutable evidence preservation, and deciding not to retry that comparison.
- Many cycles were spent trying to design a highly defensive monolithic execution capsule. Independent audits repeatedly found circular or future-dependent identities. Reports 41 through 59 produced no executable route and no result.
- The first empirical repair was declared ready too early. Independent audits then found more alias-following paths and a test environment that exposed a hidden failure. Multiple repair and audit cycles were needed.
- The first “full-route” test replaced important hash checks with accepted values for unrelated temporary bytes. Replacing that with meaningful real-byte evidence took another cycle.
- “Finish the paper” was misread as permission to replace the empirical paper with a protocol-only paper. That artifact was mechanically valid but did not satisfy the user's goal. The time spent on that conversion did not produce results.
- Governance and handoff documents were repeatedly synchronized as the project changed direction. This preserved truth, but it did not train or evaluate a model.

## Fable and old-advisor guidance

### Historical Fable guidance

The historical Fable role was a boss-level quality checker, not an implementation worker. The main instructions are in `prompts/28_fable_advisor.md`, with older operating plans in `prompts/27_ceo_replan.md`, `prompts/29_campaign_orchestrator_fresh.md`, and `prompts/30_paper_finalization_fresh.md`.

Fable advised the team to:

- keep one orchestrator responsible for the whole project;
- split implementation, independent audit, campaign operations, and paper work into separate bounded roles;
- require explicit inputs, outputs, completion criteria, verification, failure handling, and restart instructions for each task;
- never invent numbers or silently change thresholds;
- keep work event-driven rather than spending model turns polling;
- preserve evidence on disk before closing worker contexts;
- use boss-level review for major scientific or launch decisions.

Those organizational principles were useful and were largely adopted. Workers were separated from auditors, evidence reports were written, thresholds remained frozen, and later orchestration used bounded tasks.

Parts of the old Fable plan are obsolete:

- Its July 2026 submission dates have passed.
- Its mandatory two-cache PushT pilot was superseded after the Depth Anything 3 PushT route stopped and the owner authorized progress toward actual results under the recovered MapAnything contract.
- Its old pane and watcher topology was retired.
- Its protocol-first or deadline-first paper posture does not override the user's later direct order to obtain actual empirical results.

`prompts/35_codex_governance_takeover.md` later retired active Claude and Fable work and replaced it with a Codex orchestrator plus a bounded Codex advisor. The current user now explicitly wants a fully informed Fable boss pane after restart. That new Fable pane should review high-level direction and evidence, not code, dispatch workers, or independently resume work.

### Preserved old advisor

The reserved old advisor session `019f75a1-7f32-71d1-ab61-cb1a87e6e773` authorized the conventional route recorded in `reports/73_results_execution_authority.md`:

- close the empirical defects and require an independent PASS;
- abandon the recursive capsule design;
- require an independently accepted CPU/no-GPU probe before GPU work;
- begin with no more than four one-A100 jobs;
- expand only after accepted integrity, resume, memory, and throughput evidence, never above twelve;
- independently audit outcomes before paper inclusion.

That advice remains useful. The old advisor must not become the new orchestrator or a routine worker. Ask it only bounded historical questions when the fresh team cannot resolve old intent from files and code.

## Current repository state

Repository:

```text
/home/user/azirar/dinocular-wm/dino_wm
```

Current branch and accepted base:

```text
branch: feat/dinocular-encoder
HEAD:   158c74704bd18343dfbccb2fb25f4b6726c33f93
```

Current tracked diff after the interrupted final worker:

```text
17 files changed
3062 insertions
349 deletions
```

Tracked modified files:

```text
conf/study_matrix.yaml
datasets/depth_cache.py
datasets/pusht_dset.py
eval_encoder_swap.py
models/dinocular.py
p3_completion.py
tests/test_p2a_harness.py
tests/test_p3_completion.py
tools/collect_p3_completion.py
tools/collect_runs.py
tools/dinocular_container_env.sh
tools/harness_common.py
tools/make_manifests.py
tools/run_matrix_card.py
tools/submit_matrix.py
tools/submit_p3_chain.py
train.py
```

Relevant untracked candidate paths:

```text
conf/encoder/dinocular_pusht_empirical.yaml
conf/encoder/dinocular_zerodepth_pusht_empirical.yaml
contracts/depth_consumption_index_v2.yaml
contracts/pusht_mapanything_empirical_v1.json
empirical_depth_contract.py
tests/test_pusht_empirical_contract.py
```

Unrelated and outside candidate scope:

```text
.opencode/
```

No commit or push was made for the empirical candidate. Do not commit, discard, reset, or selectively revert it until the user approves a reviewed recovery plan.

Read-only state commands:

```bash
git -C /home/user/azirar/dinocular-wm/dino_wm rev-parse --abbrev-ref HEAD
git -C /home/user/azirar/dinocular-wm/dino_wm rev-parse HEAD
git -C /home/user/azirar/dinocular-wm/dino_wm status --short
git -C /home/user/azirar/dinocular-wm/dino_wm diff --stat
git -C /home/user/azirar/dinocular-wm/dino_wm diff --numstat
```

## Current experiment and HPC state

There are no queued or running Dinocular jobs. A fresh `sacct` query found no Dinocular job since 2026-07-20. No GPU is allocated to this project. No training, evaluation, or collection is active.

Scheduler check command:

```bash
ssh marvin 'squeue -u "$USER" -o "%i|%j|%T|%P|%M|%l|%R"'
```

Recent Dinocular accounting check:

```bash
ssh marvin 'sacct -X -n -P -S 2026-07-20 -u "$USER" -o JobIDRaw,JobName,State,ExitCode,Start,End | grep -i dinocular || true'
```

Five unrelated jobs are visible in the user's queue. They belong to `cf-shadows-scenes-ext`, not Dinocular. Their output locations were not inspected because the project instructions forbid touching unrelated campaign state.

| Exact job ID | Purpose visible from name | State | Output location | Check command |
|---|---|---|---|---|
| `26021266_296` | unrelated `cf-shadows-scenes-ext` task | `PENDING`, `DependencyNeverSatisfied`, `mlgpu_short` | unknown and intentionally not inspected | the `squeue` command above |
| `26021266_294` | unrelated `cf-shadows-scenes-ext` task | `PENDING`, `DependencyNeverSatisfied`, `mlgpu_short` | unknown and intentionally not inspected | the `squeue` command above |
| `26021266_257` | unrelated `cf-shadows-scenes-ext` task | `PENDING`, `DependencyNeverSatisfied`, `mlgpu_short` | unknown and intentionally not inspected | the `squeue` command above |
| `26021266_237` | unrelated `cf-shadows-scenes-ext` task | `PENDING`, `DependencyNeverSatisfied`, `mlgpu_short` | unknown and intentionally not inspected | the `squeue` command above |
| `26021266_191` | unrelated `cf-shadows-scenes-ext` task | `PENDING`, `DependencyNeverSatisfied`, `mlgpu_short` | unknown and intentionally not inspected | the `squeue` command above |

Do not cancel, modify, inspect outputs, or add dependencies to those jobs.

The stale local watcher targeting retired pane `w3:pH` was stopped. It appeared first as PID `3574041` and later as PID `10008`; both were terminated. No owner watcher should be restarted unless a newly authorized Dinocular job exists. There is no recurring scheduled task.

## Current worker and pane state

Closed after capturing useful context:

- `w3:p3H` / `w3:t3A`, empirical implementation worker, stopped with partial unverified work and no report 80.
- `w3:p3J` / `w3:t3B`, execution-route designer, completed report 81 and restart note 82.
- `w3:p3K` / `w3:t3C`, empirical independent auditor, completed report 78.
- `w3:p3M` / `w3:t3D`, execution-route independent auditor, completed report 79.
- `w3:p1X` / `w3:t1P`, historical integration-depth engineer.
- `w3:p1Y` / `w3:t1Q`, historical independent science auditor.
- `w3:p1Z` / `w3:t1R`, historical campaign execution sub-orchestrator.

Preserved only:

- `w3:p10` / `w3:t1N`, primary orchestrator.
- `w3:p1P` / `w3:t1F`, old reserved advisor.

## Evidence versus hopes

| Statement | Status |
|---|---|
| The 36-cell design and target steps are fixed | Evidence |
| The PushT MapAnything cache has accepted bytes and validation | Evidence |
| The historical SIF and source bundle have exact verified hashes | Evidence |
| Earlier empirical tests were substantially green | Evidence, but predates the last interrupted edit |
| The current task-68 edits close every materializer alias defect | Hope only; unverified and no report exists |
| Report 81 is a complete executable design | Author claim only; not independently audited |
| The current dirty tree is safe to release | Unsupported |
| A CPU probe will pass | Unknown |
| Training will fit memory and finish within projected time | Unknown until measured |
| DINOcular will outperform the baseline | Unknown |
| The complete empirical paper can be finished | Achievable goal, not current evidence |

## Main blockers and unresolved questions

1. **Interrupted implementation repair.** The current `tools/make_manifests.py` and test edits are partial. Their exact correctness and test state are unknown.
2. **No independent implementation acceptance.** Report 78 remains the latest independent implementation verdict and it is a rejection.
3. **Unaudited route design.** Report 81 has only author-side structural checks.
4. **No accepted current source release.** The immutable historical bundle does not contain the current empirical implementation.
5. **No accepted execution implementation.** The card, supervisor, network isolation, receipt, and finalizer design still needs code and independent review.
6. **No CPU/no-GPU proof.** The route must be demonstrated on the actual target before any GPU launch.
7. **No current performance measurements.** DINOcular memory and throughput remain unknown for production sizing.
8. **Compute scale.** The old projection was about `970.681` A100-hours before reserve, but it was projected from DINOv2 rates. Current DINOcular rates must be measured rather than assumed.
9. **Dirty-tree recovery choice.** A fresh team must decide, with user approval, whether to finish the partial task-68 edits or restore only that interrupted portion and reapply the repair cleanly. Do not make this choice automatically.
10. **Paper objective.** The protocol-only PDF must not be mistaken for completion again.

## Practical plan to reach the validation goal

The fresh team must wait for the user before adopting or executing this plan.

1. **Reconcile the interrupted task-68 diff.** Read report 78, inspect only the final materializer and test changes, and identify precisely which partial edits came from the stopped worker. Do not discard anything before review.
2. **Finish the narrow materializer repair.** Ensure producer decision, held-out manifest, held-out metadata, fixed evaluation manifest, and fixed evaluation metadata all receive component-wise no-alias validation before any open, hash, or parse.
3. **Finish the complete-route test.** Enter through the public `make_training()` path, produce and reload the complete 36-card matrix, select the intended PushT DINOcular card from that output, and traverse live verification, training-chain verification, real cache reader, real encoder checkpoint loading, and receipt classification.
4. **Run the exact direct, adjacent, resume, focused alias, complete-route, compile, diff, and shell checks.** Record the current counts and hashes. Prior counts are not enough.
5. **Use a fresh independent implementation auditor.** The implementer must not audit the repair. Any rejection returns to a separate bounded implementer.
6. **Independently audit report 81.** Verify that it closes all three report-79 findings and introduces no new cycle, ambiguity, missing node, or impossible prerequisite.
7. **Create an accepted clean release.** Only after implementation and route review pass, create a clean immutable source release from the reviewed candidate, excluding `.opencode`, and record exact commit, tree, bundle, SIF, test transcript, and independent acceptance identities.
8. **Implement the minimum conventional execution supervisor.** Build only what the accepted route requires: closed cards, external digests, later decisions, narrow read-only inputs, narrow writable outputs, exact network denial, post-state inventory, output manifest, supervisor receipt, and fixed scheduler-finalization behavior.
9. **Run and independently audit a CPU/no-GPU probe.** It must use the actual target Apptainer binary, exact SIF, exact no-network flags, exact mounts, no GPU, real card constructors, and no production mutation.
10. **Run serialized GPU canaries.** First geometry, then deterministic resume, then timing and memory. Audit each before proceeding.
11. **Start the exact first four one-A100 lineages only after acceptance.** These are DINOcular seed 1 for PushT, Wall, Rope, and Granular. Never exceed four concurrent jobs initially.
12. **Audit first-wave integrity and continuation.** Check identities, memory, throughput, deterministic resume, receipts, and outputs. Expand only after independent acceptance and never above twelve concurrent one-A100 jobs.
13. **Complete all 36 training cells.** Use exact target steps and paired seeds. Do not silently shorten, substitute, or tune after observing outcomes.
14. **Evaluate and independently accept all 36 cells.** Use the predeclared fixed manifests and planning procedures.
15. **Collect exactly 36 unique rows and independently audit them.** Reject duplicates, omissions, altered identities, and unaudited values.
16. **Finish the empirical paper.** Start from the intended results-oriented manuscript, insert only independently accepted numbers, retain required PushT disclosures, rebuild, and run fresh scientific and mechanical audits.

## Exact restart instructions

Do not run these commands until the user explicitly asks to restart.

### Fresh Codex 5.6 Sol high orchestrator

Create a new tab:

```bash
herdr tab create \
  --workspace w3 \
  --cwd /home/user/azirar/dinocular-wm \
  --label dinocular-results-orchestrator-restart \
  --no-focus
```

Copy the returned initial `pane_id`, then launch:

```bash
herdr pane run <NEW_ORCHESTRATOR_PANE_ID> \
  "/home/user/azirar/.local/bin/claudex --model gpt-5.6-sol --effort high -n dinocular-results-orchestrator-restart"
```

After it reaches its prompt, send exactly this role brief:

```text
You are the fresh primary orchestrator for DinocularWorldModel. Read /home/user/azirar/dinocular-wm/RESTART_HANDOFF.md in full, then read only the files it names for the first blocked decision. Do not replan, edit, spawn workers, query the scheduler, run tests, or resume experiments yet. Verify the stated repository and pane identities read-only, summarize any discrepancy, and wait for the user. Preserve separate implementer, independent-auditor, and campaign-suborchestrator roles when work is later authorized.
```

### Fully informed Fable boss pane

Create a separate tab:

```bash
herdr tab create \
  --workspace w3 \
  --cwd /home/user/azirar/dinocular-wm \
  --label dinocular-fable-boss-restart \
  --no-focus
```

Copy the returned initial `pane_id`, then launch:

```bash
herdr pane run <NEW_FABLE_PANE_ID> \
  "/home/user/azirar/.local/bin/claude --model claude-fable-5 -n dinocular-fable-boss-restart"
```

After it reaches its prompt, send exactly this role brief:

```text
You are the fully informed Fable boss reviewer for DinocularWorldModel. Read /home/user/azirar/dinocular-wm/RESTART_HANDOFF.md, prompts/28_fable_advisor.md, reports/70_owner_results_reopen.md, reports/73_results_execution_authority.md, reports/78_independent_empirical_audit_closure.md, reports/79_independent_corrected_execution_route_audit.md, reports/81_final_bounded_results_execution_route.md, and reports/82_worker_restart_note_execution_route.md. The real goal is accepted empirical results for all 36 locked cells and a completed empirical paper. Do not code, audit implementation details, dispatch workers, run commands, poll, or resume the project. Review only high-level direction and evidence boundaries, then wait for the user.
```

### Preserved old-advisor pane

Do not create a replacement. Verify and focus the existing context only when old history is unclear:

```bash
herdr pane get w3:p1P
herdr tab focus w3:t1F
```

Its preserved identity is:

```text
pane:    w3:p1P
tab:     w3:t1F
session: 019f75a1-7f32-71d1-ab61-cb1a87e6e773
```

Use it only for one bounded historical question at a time. It must not orchestrate, code, audit, dispatch, monitor, or resume work.

### Worker pattern after user approval

When the user later authorizes work, preserve this pattern:

- one high-effort orchestrator owns dependencies and evidence;
- bounded implementation workers default to `claudex` medium unless a harder task requires high effort;
- independent auditors use fresh contexts and never repair work they audit;
- scheduler operations use a separate bounded campaign sub-orchestrator and accepted cards only;
- no model polling; use event-driven waits or a watcher only after an owned job exists;
- close worker panes after their report and restart note are captured.

Do not spawn any of those roles during restart setup. The fresh orchestrator and Fable boss must wait for the user.

## First files to read after the user authorizes restart

Read in this order and stop at the first contradiction:

1. `RESTART_HANDOFF.md`
2. `reports/70_owner_results_reopen.md`
3. `reports/73_results_execution_authority.md`
4. `reports/78_independent_empirical_audit_closure.md`
5. current diff in `dino_wm/tools/make_manifests.py`
6. current diff in `dino_wm/tests/test_pusht_empirical_contract.py`
7. `reports/79_independent_corrected_execution_route_audit.md`
8. `reports/81_final_bounded_results_execution_route.md`
9. `reports/82_worker_restart_note_execution_route.md`

Treat the old top overlays in `STATUS.md` and `HANDOFF.md` as historical provenance where they conflict with the later owner reopening and this restart handoff.

CAN RESTART
