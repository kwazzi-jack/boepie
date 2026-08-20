---
type: Cab Note
title: shadeMS diagnostics
description: How shadems is used for fast visibility QA during calibration and flagging loops.
tags: [cab-note, shadems, diagnostics, qa]
managed_by: boepie
---

# shadems

`shadems` provides rapid visual diagnostics of visibility behavior across baselines, scans, and spectral structure. It is a quality-control cab used between destructive operations.

Use it over heavier bespoke plotting pipelines when you need quick operational decisions inside recipe loops. Use specialized analysis tooling only when final publications require custom statistics.

Gotchas observed in practice:

- Plot interpretation is context-sensitive; apparent structure can be physical or instrumental.
- Diagnostics must be compared before and after each major operation to avoid false conclusions.
- Ensure products are archived with run context so calibration decisions remain auditable.

# Citations

- smirnovAfricanusIVStimela22025

