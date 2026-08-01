---
type: Cab Note
title: casa.setjy
description: Role of casa.setjy in establishing calibrator flux models for transfer calibration.
tags: [cab-note, casa, setjy, flux-scale]
managed: boepie
---

# casa.setjy

`casa.setjy` sets the calibrator model basis used by downstream cross-calibration. It anchors the transfer chain to an explicit flux and polarization model context.

Use it over implicit assumptions when calibrator transfer quality matters for absolute target scaling. Skip only when an equivalent model state is already verified in your dataset branch.

Gotchas observed in practice:

- Inconsistent calibrator modeling propagates through every later gain table.
- Cross-project comparisons are unreliable when flux-model assumptions differ.
- Re-run calibrator model setup when changing calibrator selection strategy.

# Citations

- cornwellNewMethodMaking1981
- hamakerUnderstandingRadioPolarimetry1996

