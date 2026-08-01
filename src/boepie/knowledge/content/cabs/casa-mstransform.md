---
type: Cab Note
title: casa.mstransform
description: How casa.mstransform prepares analysis-ready visibility products through controlled reshaping.
tags: [cab-note, casa, mstransform, data-prep]
managed: boepie
---

# casa.mstransform

`casa.mstransform` reshapes datasets for analysis-ready products through averaging, regridding, or frame transforms. It is a practical bridge between calibrated visibilities and efficient imaging inputs.

Use it over manual preprocessing chains when you need one reproducible transformation stage in CASA. For aggressive transformations, keep a lightly transformed branch for comparison.

Gotchas observed in practice:

- Transform choices can alter calibration assumptions if applied too early or too strongly.
- Excessive averaging can mask residual issues that later appear as imaging artefacts.
- Always compare diagnostics before and after transformation.

# Citations

- smirnovRevisitingRadioInterferometer2011
