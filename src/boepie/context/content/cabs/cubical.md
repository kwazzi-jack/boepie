---
type: Cab Note
title: CubiCal gain solver
description: Where CubiCal remains useful, and tradeoffs versus QuartiCal in current workflows.
tags: [cab-note, cubical, calibration, selfcal]
managed_by: boepie
---

# cubical

`cubical` is a mature gain calibration package based on complex optimization and widely used in direction-dependent calibration workflows. It remains useful where existing data products and operational habits are tuned to its behavior.

Use it over `quartical` when you need continuity with established CubiCal-driven reduction history or package-specific tooling in your team. Use `quartical` when new scalable deployments and broad distributed execution are the primary constraint.

Gotchas observed in practice:

- Cross-tool comparability can be poor if model assumptions differ across loops.
- Migrating between CubiCal and QuartiCal inside one project should be done with controlled checkpoints.
- Keep all exact option semantics live by querying cab schema tools rather than static notes.

# Citations

- kenyonCubicalFastRadio2018
- smirnovRadioInterferometricGain2015

