---
type: Cab Note
title: casa.gaincal
description: How casa.gaincal fits into calibrator and selfcal solves and when alternatives are better.
tags: [cab-note, casa, gaincal, calibration]
managed: boepie
---

# casa.gaincal

`casa.gaincal` is a standard CASA gain solver for calibrator and iterative target workflows. It is a dependable choice in CASA-centered calibration stacks.

Use it over `quartical` or `cubical` when your project depends on CASA-native calibration table semantics and tooling. Use `quartical` or `cubical` when distributed scaling or alternative solver behavior is the core requirement.

Gotchas observed in practice:

- Solutions are highly model dependent; weak calibrator assumptions lead to unstable transfer.
- Iterative use without diagnostics can hide drift and overfitting.
- Keep solver strategy paired with imaging checkpoints.

# Citations

- pearsonImageFormationSelfCalibration1984
- cornwellNewMethodMaking1981

