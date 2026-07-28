# PushT empirical acceptance decision

## Decision

- **Empirical implementation support: ACCEPT.** Source `00e46eb0ded3b8c379ec6ebdb74c8991604f4b8f` is bound by source release SHA-256 `3b34b52f8bc07281f70f8de606fd4579f3320de0acf76977d5e97a687364cf9b`. The closed empirical implementation acceptance is SHA-256 `be96ab6a2843d99fd732cee607340bb28dc4ff42094d16cfcb5f796777a5f8bb`.
- **Immutable execution support: ACCEPT.** The closed execution acceptance is SHA-256 `297d3187b009661c44d426a894ef17122847f8162022ef04f34626190157d24d` and binds the accepted empirical runtime release SHA-256 `556e53f36cb80a5076a8269fc7dba5ce4be96d3da12c0843d94e21a7d6e05397`, the source release, and the empirical implementation acceptance.
- **Immutable probe support: ACCEPT.** The closed probe acceptance is SHA-256 `9e94cd8a97afc6ecbf049ff88ec8641ddcf65d350ffbc272dd71468ebabafc3c` and binds the accepted empirical runtime release and immutable execution acceptance.
- **Retroactive countability of the current direct PushT lineages: REJECT.** Their immutable cards declare `debug_only=true`, `not_a_campaign_run=true`, and `countable_campaign_run=false`. None of the inspected closed schemas provides a lineage-eligibility acceptance or a rule that supersedes those immutable declarations without changing lineage identity.

## Bound evidence identities

The implementation/execution/probe decision applies to authorization subject `pusht-empirical-candidate-v1`, runtime release acceptance SHA-256 `b2933fd7683dfdb956cbd8d9e86868a301783880a029adb00f089d19fe66104e`, completed checkpoint/resume jobs `26629619`, `26629620`, `26629640`, and `26629641`, and active direct jobs `26704083`, `26704084`, and `26704085` under `/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/campaign-seed1/pusht-seed1-direct-20260728h`.

## Countable relaunch requirement

A countable PushT lineage requires a new launch identity with immutable cards that declare the run countable rather than debug-only/non-campaign, while binding the accepted source release, empirical implementation acceptance, empirical runtime release, immutable execution acceptance, and immutable probe acceptance. Existing cards, checkpoints, and running jobs remain unchanged.
