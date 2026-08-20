---
type: Cab Note
title: QuartiCal gain solver
description: How QuartiCal fits into high-scale calibration loops and when to choose it over CubiCal.
tags: [cab-note, quartical, calibration, gains]
managed_by: boepie
---

# quartical

`quartical` solves interferometric gains with a design optimized for scale and distributed execution. It is commonly used as the default direction-independent calibration engine in modern Stimela workflows.

Use it over `cubical` when you need scalable cloud or cluster execution and flexible gain-chain workflows. Use `cubical` when your existing reduction stack is already validated around CubiCal behavior and you need strict continuity with prior products.

Gotchas observed in practice:

- Calibration quality is model-limited; unstable or biased sky models produce formally converged but scientifically poor gain updates.
- Aggressive looping can overfit weak structure; evaluate residual behavior at each iteration.
- Pull current parameter details from live schema tools, because available controls evolve with package versions.

# Citations

- kenyonAfricanusIIQuartiCal2025
- smirnovRadioInterferometricGain2015
- salviniFastGainCalibration2014

