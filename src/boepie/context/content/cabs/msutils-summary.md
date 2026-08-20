---
type: Cab Note
title: msutils.summary
description: Why msutils.summary is a preflight step and what it catches before expensive runs.
tags: [cab-note, msutils, summary, preflight]
managed_by: boepie
---

# msutils.summary

`msutils.summary` reports key measurement set structure and is commonly run as a preflight guardrail. It catches obvious scope and content mismatches before long calibration or imaging jobs.

Use it over jumping straight into solver cabs when working with unfamiliar or newly split datasets. Use `casa.listobs` in parallel workflows where CASA-native summaries are already standard.

Gotchas observed in practice:

- Summary output is descriptive, not corrective; downstream decisions still need explicit policy.
- Teams often skip this on repeated runs and miss accidental dataset swaps.
- Keep preflight summaries with run logs for reproducibility.

# Citations

- smirnovAfricanusIVStimela22025

