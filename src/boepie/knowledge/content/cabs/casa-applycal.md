---
type: Cab Note
title: casa.applycal
description: How casa.applycal applies calibration products and what to verify immediately after transfer.
tags: [cab-note, casa, applycal, transfer]
managed: boepie
---

# casa.applycal

`casa.applycal` applies calibration products to datasets and is the pivot from calibration tables to corrected visibilities. It is central in both cross-calibration and some selfcal branches.

Use it over ad hoc visibility correction paths when you need consistent CASA table application semantics. In mixed ecosystems, ensure downstream cabs consume the intended corrected state.

Gotchas observed in practice:

- Applying the wrong calibration product set can quietly contaminate all later products.
- Immediate post-application diagnostics are essential before splitting or imaging.
- Keep clear lineage of corrected datasets in recipe logs.

# Citations

- pearsonImageFormationSelfCalibration1984

