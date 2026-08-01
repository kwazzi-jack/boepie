---
type: Cab Note
title: taql.update table edits
description: When taql.update is the right tool for surgical edits and how to avoid broad unintended changes.
tags: [cab-note, taql, table-edits, measurement-set]
managed: boepie
---

# taql.update

`taql.update` applies direct table updates and is best for narrow, explicit fixes that are awkward in higher-level flaggers. It is a surgical instrument, not a broad reduction strategy.

Use it over `casa.flagdata` only when you need exact table logic that maps directly to known defects. Prefer higher-level tools for routine operations where intent and auditability matter more than expression flexibility.

Gotchas observed in practice:

- Broad selection logic can damage large portions of a dataset quickly.
- Every edit should be paired with a checkpoint and post-edit diagnostics.
- Keep update expressions in recipe context so changes are reproducible.

# Citations

- smirnovAfricanusIVStimela22025

