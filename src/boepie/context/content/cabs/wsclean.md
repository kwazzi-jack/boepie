---
type: Cab Note
title: wsclean imager
description: When to use wsclean for fast wide-field deconvolution and what can go wrong in difficult fields.
tags: [cab-note, wsclean, imaging, deconvolution]
managed_by: boepie
---

# wsclean

`wsclean` is the workhorse continuum imager for most Stimela pipelines. It combines fast wide-field imaging with practical deconvolution modes that are robust on large modern datasets.

Use it over `pfb-imaging` when you need fast, dependable imaging turnaround and your field can be modeled well by standard deconvolution families. Prefer `pfb-imaging` when diffuse structure or reconstruction goals require a different optimization framework.

Gotchas observed in practice:

- Off-axis bright sources can dominate residual artefacts unless calibration and masking strategy are staged carefully.
- Spectral and wide-field behavior can bias faint structure if deconvolution choices are not matched to field complexity.
- Solver controls are numerous; retrieve current semantics from live cab schema tools rather than static notes.

# Citations

- offringaWscleanImplementationFast2014

