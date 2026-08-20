---
type: Cab Note
title: Tricolour flagger
description: How Tricolour is used as primary automated RFI rejection before calibration.
tags: [cab-note, tricolour, flagging, rfi]
managed_by: boepie
---

# tricolour

`tricolour` is a statistical visibility flagger used as the first automated RFI pass in many workflows. It is effective as an early-stage cleaner before gain solving and deep imaging.

Use it over `casa.flagdata` for initial broad automated rejection on large datasets. Use `casa.flagdata` after Tricolour for targeted cleanup decisions tied to specific observational regimes.

Gotchas observed in practice:

- Over-flagging can silently reduce calibration leverage if not checked against coverage diagnostics.
- Under-flagging leaves coherent outliers that contaminate gain solves and deconvolution.
- Always inspect resulting data health with diagnostics cabs before proceeding.

# Citations

- mitchellRealTimeCalibrationMurchison2008

