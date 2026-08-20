---
type: Cab Note
title: msutils.copycol
description: How msutils.copycol supports reversible calibration loops through explicit column snapshots.
tags: [cab-note, msutils, measurement-set, rollback]
managed_by: boepie
---

# msutils.copycol

`msutils.copycol` copies visibility columns inside a measurement set and is used for rollback checkpoints. It is a practical safeguard before aggressive calibration or flag edits.

Use it over irreversible in-place workflows when iteration strategy is uncertain. If storage is constrained, prefer milestone checkpoints only, but keep at least one recoverable baseline state.

Gotchas observed in practice:

- Copy operations can preserve bad states if performed after contamination has already spread.
- Column lineage must be tracked in recipe logs or rollback becomes ambiguous.
- Coordinate with downstream cabs so they read the intended working column.

# Citations

- smirnovAfricanusIVStimela22025

