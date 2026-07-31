# Paper track status

Last updated: 2026-07-18 (PP22 independently audited submission freeze complete)

## PP22 AUDITED SUBMISSION FREEZE 2026-07-18

- Final state: **FROZEN AND OWNER-UPLOAD-READY; NOT EXTERNALLY SUBMITTED.** No OpenReview upload or external message has occurred.
- Read-only PaperPilot reconfirmation: project `6a5662553c2ec1b86de59f76`, live revision `fork-34b3737794bdee49`, empty author diff with `changedPaths: []`. Frozen `main.tex` SHA-256 is `67221fa95993d17e8f3b8fcb5c3628df829c80354f28c2da10acf35b077d39e8`; frozen `main.bib` SHA-256 is `d3c58a974bdd106b20899fb5efe01fae2a2e0247907b2b0bc5d46ecb865cb792`.
- Upload exactly `reports/ECCV_SUBMISSION_DRAFT.pdf`, SHA-256 `0ad346391a67532421548868c63b08a7f8564666c19bf9531ab23e152a1d4af5`. It is 7 pages, references begin on page 6, anonymization passes, and it contains exactly 67 visible TBD markers.
- No discrepancy appeared, so no recompilation was performed. The 67 TBD markers, unresolved producer choice, lack of figures, and six cosmetic overfull hboxes remain honestly disclosed limitations.
- Old PaperPilot approval hashes and optional bibliography/prose publication steps are retired. Owner steps are in `reports/11_submission_checklist.md`; full freeze evidence is in `reports/pp22_submission_freeze.md`.
- No further manuscript edit, training, or watcher is planned without a new owner order.

## PP20 OWNER-AUTHORIZED PUBLICATION 2026-07-18

- Final verdict: **PUBLISHED**. The byte-identical PP18 R1 to R4 candidate passed the repaired complete checker before publication and again after publication/PDF installation. Both runs used `PP13_EXPECTED_TBD_COUNT=67` and returned nine PASS lines with exact source/PDF counts `67/67`.
- PaperPilot project `6a5662553c2ec1b86de59f76` advanced from revision `fork-04418a1dc1ea5656` to `fork-34b3737794bdee49`. Publish pushed only `main.tex`, created no file, and reported no conflict. Post-publish sync was flushed with `conflicts: []`; author diff is empty.
- Final `main.tex` SHA-256 is `67221fa95993d17e8f3b8fcb5c3628df829c80354f28c2da10acf35b077d39e8`. Final `main.bib` SHA-256 remains `d3c58a974bdd106b20899fb5efe01fae2a2e0247907b2b0bc5d46ecb865cb792`.
- Final `reports/ECCV_SUBMISSION_DRAFT.pdf` SHA-256 is `0ad346391a67532421548868c63b08a7f8564666c19bf9531ab23e152a1d4af5`, exactly matching the separately verified isolated Willow build. It is 7 pages; references start on page 6; anonymization and bibliography integrity pass.
- Published content is exactly R1, R2, R3, and R4. R5 and R6 remain unchanged. No MapAnything prose was added. Every remaining unmeasured outcome remains TBD, and all three producer-choice TBD markers remain explicit.
- Immutable PP18 first-attempt record remains SHA-256 `09a03e8430da61c44b1f353f802bd6f37400aa196d57f3081307150258a7f8ee`. Full PP20 evidence: `reports/pp20_paperpilot_authorized_publish.md`.

## PP18 OWNER-AUTHORIZED UPDATE 2026-07-18

- Owner authorization covered exact proposal hunks R1, R2, R3, and R4 only. R5 and R6 remained unchanged. No unmeasured outcome was replaced.
- Pre-edit PaperPilot verification passed at project `6a5662553c2ec1b86de59f76`, revision `fork-04418a1dc1ea5656`, `main.tex` SHA-256 `df83b6d7d669ed1d017a2544bf144e8bf7227378dd9c3652306d870b891e0473`, `main.bib` SHA-256 `d3c58a974bdd106b20899fb5efe01fae2a2e0247907b2b0bc5d46ecb865cb792`, and empty author diff.
- The exact R1 to R4 candidate compiled on Willow in an isolated directory with all four LaTeX/BibTeX stages exiting zero. It remained seven pages, references began on page 6, anonymization and bibliography checks passed, and source/PDF both contained 67 TBD markers.
- Publication verdict: **NOT PUBLISHED**. The established final checker returned 8 PASS and 1 FAIL because its TBD drift gate expected the pre-R4 count of 79 rather than the authorized post-R4 count of 67. The fail-closed rule prohibited publication.
- The author bridge was restored byte-for-byte. Live revision remains `fork-04418a1dc1ea5656`, live source hashes remain unchanged, author diff is empty, and `reports/ECCV_SUBMISSION_DRAFT.pdf` remains unchanged at SHA-256 `dcc093589dd5bb476440b34b033687dcdc78736ffbd426030a01ee491869d0c9`, 7 pages.
- MapAnything cache acceptance was reconciled but not inserted because the source has no exact mechanical cache-status placeholder and no head-to-head pilot result exists. Producer choice remains TBD. No claim says or implies that Depth Anything 3 lost scientifically.
- Full bounded evidence, exact reverted candidate diff, hashes, build output, checker result, and verdict: `reports/pp18_owner_authorized_update.md`.

## FRESH TAKEOVER 2026-07-17 (pane w3:p1D, paper-finalization)

Fresh paper-finalization orchestrator took over the PAPER track (P1 to P5) from disk only. Acting CEO is
w3:p1B. Codex is locked until 2026-07-24, so Opus is the primary editor and GLM 5.2 via oc-offload does the
P4 cross-review. Touched no cluster jobs, no main-campaign state files. Read HANDOFF sections 1,2,4,5,6,7,7b,
this file, `prompts/pp01_paper_orchestrator.md`, and `reports/paper_pending_approval_20260717.md` only.

### State verified this turn (read-only)
- Fallback PDF is intact and upload-ready. `reports/ECCV_SUBMISSION_DRAFT.pdf` SHA-256
  `dcc093589dd5bb476440b34b033687dcdc78736ffbd426030a01ee491869d0c9`, 7 pages (pdfinfo). Matches the
  checkpoint hash exactly. If no verdict and no new number lands, this ships on 2026-07-20.
- P1 deliverables all present and consistent on disk. `reports/pp10_tbd_ledger.md` (`f4dfc6b2b0212ae7...`),
  `reports/pp11_accepted_numbers.md` (`26910045806004db...`), `reports/pp12_cross_review.md`
  (`ecd959b22be8dac7...`), `reports/pp09_source_access.md` (`f6b01b44e82f74f7...`). P1 is DONE.
- Eligible numbers today are exactly R1 to R3 (methods-level Wall DA3 cache, exact training-window counts,
  PushT release total). No P2a, P3, P4, or P5 OUTCOME cell is eligible. Accepted outcomes remain zero.
- No boss verdict on open question #2 (R1 to R6) exists. No `reports/BOSS_INBOX.md` existed before this turn.
- `scratch/pp13_final_check.sh` compile gate cannot run right now: the PaperPilot bridge mount
  `~/.local/share/paperpilot-bridge/mounts/6a5662553c2ec1b86de59f76/` is empty (not mounted). This blocks a
  FRESH rebuild-and-check only. It does NOT affect the already-built fallback PDF, whose hash and page count
  are independently verified above. Remount is only needed if a rebuild is authorized (boss approves a hunk).

### Gate status of P1 to P5
- P1 paper decision and ledger: DONE. Ledgers, accepted-number list, and TBD classification exist and agree.
- P2 mechanical fold-in and build: BLOCKED on boss approval. R1 to R3 are prose additions (new sentences),
  not mechanical TBD-cell substitutions, so they wait for a red or green. No accepted OUTCOME to fold.
- P3 final mechanical pass: prior evidence is 9 PASS (`reports/pp13_final_pass_checklist.md`). A fresh run is
  deferred until a rebuild is authorized and the bridge is remounted. Fallback PDF already carries this pass.
- P4 different-model cross-review: prior GLM 5.2 PASS covers the applied H1 to H11 baseline
  (`reports/pp12_cross_review.md`). A new cross-review is required only when a new content diff is produced.
- P5 freeze: NOT executed. Freeze is due 2026-07-19 evening, not today. Freezing now would foreclose folding
  numbers that could still land before the deadline. Held open per HANDOFF section 1.

### Actions taken this turn
- Wrote `reports/BOSS_INBOX.md` with the concise open-question-#2 decision request and the per-hunk paper-track
  recommendation (approve R1 to R4 conditional on 7 pages, reject R5.1, approve R5.2 to R5.7, retain R6 markers).
- No manuscript edit, no publish, no cluster action, no invented number. TBD stays TBD.

### What wakes this pane (event-driven, no polling)
1. A boss verdict on open question #2 arrives (in-pane, BOSS_INBOX reply, or via acting CEO w3:p1B). Then run
   P2 fold of approved hunks, remount bridge, rebuild on willow, rerun P3 checker, P4 cross-review the diff.
2. A new accepted, hash-bound, paper-eligible OUTCOME number lands in `STATUS.md` (main campaign owns it).
   Then fold it as a mechanical substitution, rebuild, and re-verify.
3. The 2026-07-19 evening freeze window with no approved change and no new number. Then P5 freezes and ships
   the existing verified PDF, and prints PAPER-COMPLETE with hash `dcc09358...1869d0c9`.

## HANDOFF: read this block first

A clean-break replan was ordered on 2026-07-17. This section is the complete paper-track state so the
incoming CEO planner can take over without me or the boss. All paper workers are stopped and their tabs
closed. The only live paper pane is this orchestrator `w3:pV`, now waiting for the re-brief. I touched no
other campaign's panes. Panes `w3:p12`, `w3:p13`, `w3:p14`, `w3:p16` and `w3:pT` belong to the MAIN
campaign orchestrator, not to the paper track, and were left alone.

### Where the paper stands right now
- The submission PDF is already upload-ready as it stands. `reports/ECCV_SUBMISSION_DRAFT.pdf`, SHA-256
  `dcc093589dd5bb476440b34b033687dcdc78736ffbd426030a01ee491869d0c9`, 7 pages, references from page 6.
  If nothing else is approved, the boss can submit this file. Deadline is 2026-07-20 20:00 UTC (22:00 CEST),
  OpenReview WMEAI. Manual steps for the boss are in `reports/11_submission_checklist.md`.
- Live source is verified. PaperPilot project `6a5662553c2ec1b86de59f76` (`dinocular-wm-eccv`), revision
  `fork-04418a1dc1ea5656`, `main.tex` SHA-256 `df83b6d7d669ed1d017a2544bf144e8bf7227378dd9c3652306d870b891e0473`,
  `main.bib` SHA-256 `d3c58a974bdd106b20899fb5efe01fae2a2e0247907b2b0bc5d46ecb865cb792`, empty author diff.
  Source access is read-write through the bridge but NOTHING was published this session.
- The mechanical final-pass checker `scratch/pp13_final_check.sh` reports 9 PASS, 0 FAIL. Evidence
  `reports/pp13_final_pass_checklist.md`. The Jul 18-19 pass is now mechanical.

### Open items the planner must decide (nothing is applied)
1. The reboot proposal `reports/paper_pending_approval_20260717.md` (SHA-256
   `f1260130a8f9bb5aa0769686c089ba85169941a922f8b9052def67062b1848e3`) is drafted and UNAPPLIED. It holds
   hunks R1 to R6. Boss approval is required before any prose or table change ships. My recommendations:
   - R4 (remove the undefined `Horizon AUC` column, 12 TBD cells): approve outright. It is a real defect,
     an undefined metric with no live claim or decision-rule dependency.
   - R1 (Wall DA3 cache claim), R2 (exact window counts), R3 (PushT release total): approve subject to the
     rebuild staying within 7 pages. All three are accepted methods-level numbers, not outcome cells.
   - R5.1 (cut the abstract "all outcomes TBD" sentence): I recommend REJECT. The page gate already passes,
     so no space is needed, and that sentence is the abstract's only honest signal that the paper ships 60
     TBD cells. R5.4 to R5.7 are safe redundant cuts.
   - R6: retain all three producer-choice TBD markers. No job can produce an accepted producer name before
     the freeze, and hiding the markers would mask an open scientific choice.
2. No result-table OUTCOME value is eligible today. Accepted P2a, P3, P4, P5 outcomes are all zero. All 79
   live TBDs are unmeasured; 60 stay-TBD, 19 removable. Full per-cell ledger `reports/pp10_tbd_ledger.md`.
3. The producer pilot is unresolved. Wall DA3 cache is VALID (job `26548609`); Rope and Granular DA3 caches
   FAILED and are being refixed by the MAIN campaign, not the paper track. Watch `STATUS.md` for new
   accepted numbers with a paper target.

### What I did NOT do, on purpose
- Did not publish, stage, or edit the manuscript. Read-only all session.
- Did not touch `reports/paper_pending_approval.md` (the immutable 2026-07-16 record, SHA-256
  `fa394c8f08f7f8872e47401da8de35d591dd8317b937ded2fb66a0d4f1e57c27`, verified intact). The new proposal is a
  separate dated file.
- Did not touch `STATUS.md`, `MASTER_PLAN.md`, `BUDGET.md`, cluster jobs, or any other campaign.

## Current milestone

Speed reboot 2026-07-17, then clean-break handoff the same day. The paper track was activated early by boss
order (`prompts/26_speed_reboot.md` PAPER section) and driven by a Claude Opus sub-orchestrator in pane
`w3:pV`. The reboot fleet of six workers ran to completion, produced the reports and proposal below, then was
stopped for the clean-break replan. The results watcher is stopped, not running.

## Reboot findings (2026-07-17)

- Source access is PASS and every binding claim in this file was reverified against live state. Revision `fork-04418a1dc1ea5656`, `main.tex` SHA-256 `df83b6d7...891e0473`, empty author diff, PDF SHA-256 `dcc09358...1869d0c9`, 7 pages, exactly 79 literal TBD markers. Evidence `reports/pp09_source_access.md`.
- The reboot order's claim "Wall DA3 cache VALID with its gate record" is CONFIRMED, and it supersedes the last watcher entry. At reboot the newest paper-track evidence was the 2026-07-16 hard FAIL of validator `26547679`, so the claim was checked rather than assumed. The current truth is that Wall manifest `54bab01a-8c49-4df5-a1d9-3a76e32e5234` is VALID under Amendment `2026-07-16c` through external validation job `26548609`, which exited `0:0` with overall PASS. Receipt SHA-256 `b2404b80...122fab4`. The superseded `26547679` failure remains failure evidence and is not a paper number. Evidence `reports/pp11_accepted_numbers.md`.
- The corrected PushT release statistics are CONFIRMED accepted. The live manuscript already carries every split value correctly. Only the combined total and the exact window counts are absent.
- Eligible today: the Wall validation record, the exact released training-window counts (PushT `1,981,721`, Wall `70,848`, Rope `17,100`, Granular `17,100`), and the PushT release total (`18,706` trajectories, `2,339,250` frames). All are methods-level. NO result-table outcome cell is eligible. Accepted P2a, P3, P4, and P5 outcomes remain at zero.
- The 2026-07-16 cross-review never produced a report, so it was redone rather than waited on. `reports/pp12_cross_review.md` returns PASS on all six dimensions for the applied H1 to H11 diff. Fidelity, scope, numbers, truth, style, and integrity all pass, with no required fix.
- The 79 TBDs are now fully classified in `reports/pp10_tbd_ledger.md`. None is awaiting a job that can plausibly land before the freeze. 60 are `stays-TBD` and 19 are `removable`. The whole `Horizon AUC` column is removable and is also an undefined metric in the live text. No arm or environment row is removable.
- The mechanical final-pass checker exists at `scratch/pp13_final_check.sh` with 9 gates. Eight pass. Its TBD gate was written as a zero gate, which wrongly makes any shipped TBD an upload blocker and contradicts pp01. It is being corrected to a count-and-compare drift gate.
- The persistent results watcher was dead and is relaunched as `pp-results-watch`.

## Approval gate open (2026-07-17)

- The reboot proposal is `reports/paper_pending_approval_20260717.md`, SHA-256 `f1260130a8f9bb5aa0769686c089ba85169941a922f8b9052def67062b1848e3`. It is bound to revision `fork-04418a1dc1ea5656` and `main.tex` SHA-256 `df83b6d7...891e0473` with an empty author diff.
- Hunks are R1 Wall cache claim, R2 exact window counts, R3 PushT release total, R4 remove the undefined `Horizon AUC` column, R5.1 to R5.7 remove seven status-only prose markers, and R6 the producer-marker decision.
- NOTHING is applied. No prose change ships without boss approval. That rule is binding under pp01 and is explicitly preserved by the speed reboot.
- Orchestrator dissent on R5.1. The worker recommends approving all of R5. I recommend REJECTING R5.1 and keeping the abstract sentence "All quantitative outcomes remain TBD." The page-limit gate already passes, so no space is needed. That sentence is the only place the abstract tells a reviewer the paper carries no outcomes, and this manuscript ships 60 TBD cells. Cutting it buys about one line and costs the reader the honest framing. R5.4 to R5.7 are genuinely redundant with visible cells and captions, so they are safe.
- Recommendation on the rest: approve R4 outright, since the column is an undefined metric and a real defect. Approve R1 to R3 subject to the rebuild staying within seven pages.

## Upload readiness (2026-07-17)

- The current PDF is already submission-ready. The corrected mechanical checker `scratch/pp13_final_check.sh` reports nine PASS gates and zero FAIL. Evidence `reports/pp13_final_pass_checklist.md`.
- The TBD gate was corrected at the root. It was written as a zero gate, which contradicted the pp01 rule that unmeasured cells may ship as TBD. It is now a count-and-compare drift gate. Source 79, PDF 79, expected 79, all agreeing. The expected count drops to 60 only if the boss approves every R4 and R5 removal.
- If nothing further is approved and no new result lands, the boss can submit the existing `reports/ECCV_SUBMISSION_DRAFT.pdf` as is. The Jul 18-19 pass is now mechanical.

## Results watcher (2026-07-17): STOPPED for clean break

- STATE: stopped. Pane `w3:p11` closed, no watcher process running, no v2 log left on disk. If the incoming
  planner wants a watcher, relaunch from brief `prompts/pp17_results_watch_v2.md` (v2, correct design).
- The v1 watcher was killed at PID `1038580` and its design rejected. It let a shell grep heuristic decide
  eligibility and print the sentinel. It matched the words "accepted" and "PASS" anywhere in `STATUS.md`,
  reported a timestamp fragment as the accepted number, and emitted a false eligible delta within minutes.
  Its log is quarantined at `scratch/pp14_results_watch_INVALID_20260717.md` and is not truth.
- The v2 brief fixes the split of labour at the root. The process decides only whether bytes changed. The
  model reads the actual diff, decides eligibility, and is the only thing allowed to print `PAPER-DELTA`.
  v2 ran briefly, self-corrected a baseline-representation bug, blocked correctly with no false entry, then
  was stopped for the handoff. No eligible delta was ever emitted by v2.

## Worker ledger (2026-07-17 reboot): ALL STOPPED, tabs closed

Every paper worker below completed its deliverable and its tab is now closed for the clean break. Outputs
persist on disk. Briefs persist under `prompts/` and can be re-dispatched by the incoming planner.

- `pp-source-access`, was pane `w3:pX`, codex sol medium. Brief `prompts/pp09_source_access_reverify.md`. Re-verified live revision and hashes. Output `reports/pp09_source_access.md`. Result PASS. CLOSED.
- `pp-tbd-ledger`, was pane `w3:pY`, codex sol high. Brief `prompts/pp10_tbd_ledger.md`. Classified all 79 TBDs. Output `reports/pp10_tbd_ledger.md`. CLOSED.
- `pp-accepted-numbers`, was pane `w3:pZ`, codex sol high. Brief `prompts/pp11_accepted_numbers.md`. Harvested eligible numbers. Output `reports/pp11_accepted_numbers.md`. CLOSED.
- `pp-final-checklist`, was pane `w3:p0`, codex sol medium. Brief `prompts/pp13_final_pass_checklist.md`. Built the checker. Outputs `scratch/pp13_final_check.sh` and `reports/pp13_final_pass_checklist.md`. CLOSED.
- `pp-cross-review`, was pane `w3:p15`, OpenCode GLM 5.2 via `oc-offload`, a different model from the codex author of the diff. Brief `prompts/pp12_cross_review_glm.md`. Output `reports/pp12_cross_review.md`. Result PASS on all six dimensions. CLOSED.
- `pp-proposal`, was pane `w3:p17`, codex sol high. Brief `prompts/pp15_reboot_proposal.md`. Drafted the R1 to R6 proposal. Output `reports/paper_pending_approval_20260717.md`. CLOSED.
- `pp-fix-tbd-gate`, was pane `w3:p18`, codex sol medium. Brief `prompts/pp16_fix_tbd_gate.md`. Corrected the TBD gate to a drift gate. Updated `scratch/pp13_final_check.sh` and `reports/pp13_final_pass_checklist.md`. CLOSED.
- `pp-results-watch-v2`, was pane `w3:p11`, codex luna low. Brief `prompts/pp17_results_watch_v2.md`. STOPPED and closed for the clean break, see the watcher section above.

## Prior state carried forward (2026-07-16)

## Binding state

- Approved comparison baseline is the 2026-07-15 source at revision `fork-ad8143cfb8da6330`.
- Boss-approved prose hunks H1 to H11 are applied. H6 passed its conditional page gate.
- No measured result has yet been accepted for insertion by this track.
- PaperPilot source is published at live revision `fork-04418a1dc1ea5656`.
- The verified bibliography is already present in the live source. There is no staged author diff.
- The stale approval hash in `reports/11_submission_checklist.md` must not be used.

## Worker ledger

- `pp-source-access`, pane `w3:pB`, completed with PASS. Evidence is in `reports/pp02_source_access.md`.
- `pp-truth-audit`, pane `w3:pC`, completed. Evidence is in `reports/pp03_campaign_truth_audit.md`.
- `pp-proposal`, pane `w3:pD`, completed. Proposal is in `reports/paper_pending_approval.md`.
- `pp-results-watch`, pane `w3:pE`, active persistent monitor. Delta log is `reports/pp05_results_watch.md`.
- `pp-draft-audit`, pane `w3:pF`, completed. Evidence is in `reports/pp06_draft_audit.md`.
- `pp-apply-build`, pane `w3:pM`, completed. Evidence is in `reports/pp07_apply_build.md`.
- `pp-cross-review`, pane `w3:pN`, active different-model full-diff review through OpenCode GLM 5.2.

## Evidence disposition

- Accepted P2a outcome cells: 0.
- Accepted P3 outcome cells: 0 of 36.
- Safe mechanical paper edits now: none.
- Required protocol, budget, contract, limitation, and table corrections were approved and applied.

## Approval gate

- Proposal base revision is `fork-ad8143cfb8da6330`.
- Proposal base `main.tex` SHA-256 is `711d5ad506fe1d7e6d21e9e7537f07f3e2b87e74b095de0b00813dd31669dabb`.
- Proposal file SHA-256 is `fa394c8f08f7f8872e47401da8de35d591dd8317b937ded2fb66a0d4f1e57c27`.
- Required hunks are H1 to H5 and H7 to H11. H6 is optional if page pressure requires omitting release counts.
- Application completed only after the bound revision, source hash, empty author diff, and proposal hash passed.

## Boss approval

- Approved nonoptional hunks are H1 to H5 and H7 to H11.
- H6 may be kept only if the compiled body remains within the seven-page limit.
- No outcome value is approved. Unsupported outcomes remain TBD.
- Approval is bound to revision `fork-ad8143cfb8da6330` and base `main.tex` SHA-256 `711d5ad506fe1d7e6d21e9e7537f07f3e2b87e74b095de0b00813dd31669dabb`.
- The approval proposal remains the immutable record with SHA-256 `fa394c8f08f7f8872e47401da8de35d591dd8317b937ded2fb66a0d4f1e57c27`.

## Applied source and build

- Final live revision is `fork-04418a1dc1ea5656` with an empty author diff.
- Final `main.tex` SHA-256 is `df83b6d7d669ed1d017a2544bf144e8bf7227378dd9c3652306d870b891e0473`.
- Final `main.bib` SHA-256 is `d3c58a974bdd106b20899fb5efe01fae2a2e0247907b2b0bc5d46ecb865cb792`.
- H1 to H5 and H7 to H11 were applied exactly. H6 was retained after a clean conditional compile.
- Final PDF is `reports/ECCV_SUBMISSION_DRAFT.pdf` with SHA-256 `dcc093589dd5bb476440b34b033687dcdc78736ffbd426030a01ee491869d0c9`.
- The PDF has seven pages. References begin on page 6. Body Table 2 ends on page 7.
- All four compile stages exited zero. Nine bibliography entries resolve. Anonymization passed.
- Source and PDF each retain 79 literal TBD markers. No measured outcome value was inserted.
- The source has no em dash, semicolon, `\\emph`, or unapproved prose colon.
- Seven overfull hbox warnings remain. They are recorded in `reports/pp07_apply_build.md`. No unapproved layout rewrite was made.
- The complete approved-base diff is `reports/pp07_full_diff.patch` with SHA-256 `33adf2a1ff68e0383e6ff6d60ef2b9762c366a5b78a24cb172c044260782a425`.
