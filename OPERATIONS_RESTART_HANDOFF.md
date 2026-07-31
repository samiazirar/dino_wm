# Dinocular operations restart handoff

Status: readiness only. Claude capacity is paused. Do not dispatch, reopen,
replace, close, or create any execution role until Sami explicitly resumes.

## Preserved state

- Shared/root checkout: keep untouched. It still contains the user's original
  dirty empirical candidate.
- Isolated candidate worktree:
  `/home/user/azirar/dinocular-wm-worktrees/task68-materializer`
  on branch `task/task68-materializer`.
- Clean candidate identity:
  `cdd35bda619993d1d2a7945f841852a276ca7806`.
  Last verified worktree status was clean.
- Maya context, `Worker · Maya PushT GPU Debug`, native session
  `485c24f0-fb78-4dbf-b829-69ae75ca58be`: completed and preserved. Reuse it
  only if a later observed runtime failure requires a source repair.
- Marcus context, `Suborch · Marcus Marvin Campaign`, native session
  `f7bc5de0-8c46-4779-b9dd-db2c81bc8667`: completed and preserved. Reuse it
  first to finish staging and relaunch; do not create a replacement.
- First real A100 route attempt: Marvin job `26627335`, `sgpu_short`, failed
  after 42 seconds with exit `2:0`. Complete observed error:
  `HARNESS CONTRACT FAILURE: live source commit differs for
  p2a-pusht-dinocular-s1-mapanything_recovered_framewise`.
- The exact candidate Git bundle was transferred to the isolated Marvin debug
  lineage with a matching bundle hash. Remote clone creation began but its
  completion, exact final source path, and clean status were not captured
  before capacity stopped the contexts.
- No replacement job was confirmed after `26627335`.
- No Fable advisor is open. Ask no Fable question and create none.

## Exact first bounded sequence after capacity returns

1. Resume Marcus's preserved context only.
2. Reload Marcus's final tool output and the existing isolated debug lineage;
   determine whether the already-started remote clone completed. Do not search
   broadly or create another staging lineage.
3. Finish that same clone if needed. Record its exact remote path, prove its
   `HEAD` is exactly `cdd35bda619993d1d2a7945f841852a276ca7806`, and prove
   its Git status is clean.
4. Starting from the byte-identical failed card, update or regenerate only the
   source binding required to point to that exact staged source. Preserve every
   other card field unchanged. Record the new card SHA-256 and a field-level
   comparison against the failed card.
5. Marcus immediately submits one replacement A100 `sgpu_short` job in the
   same isolated debug family. Record job ID, observed scheduler state,
   standard-output path, standard-error path, run directory, and card hash.
6. Attach the single event-driven watcher pattern to that job. Route the
   compact submission envelope to Owen and Hannah.
7. If the job fails, preserve the complete observed log. Resume Maya's existing
   context only when that log demonstrates a source-code repair is needed.
   Make no predicted placeholder or authorization repair in advance.

## Completion checks for the first bounded sequence

- Root checkout is byte-for-byte untouched by the restart.
- Local isolated worktree is clean at the exact candidate commit above.
- Remote staged clone exists at one recorded path, has the same exact commit,
  and is clean.
- Replacement card differs from the failed card only in the necessary source
  binding; all other fields compare equal.
- Replacement card hash is recorded.
- Exactly one replacement A100 `sgpu_short` job is submitted.
- Job ID, state, output/log paths, run directory, and card hash are routed to
  Hannah.
- Observation is event-driven; there is no model-turn scheduler polling.
- No new worker, suborchestrator, auditor, strategist, planner, or substitute
  task is created.

Stop after recording and routing the replacement submission. Continue only on
its material start or terminal event.
