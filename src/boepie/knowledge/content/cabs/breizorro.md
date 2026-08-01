---
type: Cab Note
title: Breizorro masking tool
description: When Breizorro masks improve deconvolution stability and when they can hide real emission.
tags: [cab-note, breizorro, masking, imaging]
managed: boepie
---

# breizorro

`breizorro` is used to generate and manipulate masks that guide deconvolution in imaging loops. It improves stability when sidelobes or complex source structure cause unbounded clean behavior.

Use it over ad hoc manual mask editing when you need reproducible, recipe-level mask generation. Use no explicit masking only when field complexity is low and deconvolution remains stable without it.

Gotchas observed in practice:

- Over-restrictive masks can block real low-surface-brightness recovery.
- Loose masks can admit noise peaks and destabilize deep clean iterations.
- Recompute masks when calibration state changes materially between imaging rounds.

# Citations

- offringaOptimizedAlgorithmMultiscale2017

