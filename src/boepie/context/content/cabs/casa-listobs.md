---
type: Cab Note
title: casa.listobs
description: Why casa.listobs is used as a scan-structure preflight in CASA-centered workflows.
tags: [cab-note, casa, listobs, preflight]
managed_by: boepie
---

# casa.listobs

`casa.listobs` summarizes observation structure in CASA terms and is frequently used before cross-calibration. It helps verify field segmentation and scan logic before solving.

Use it over generic summaries when your calibration branch is CASA-first. In mixed pipelines, pair it with `msutils.summary` for a broader preflight view.

Gotchas observed in practice:

- A clean listing does not guarantee a suitable calibrator model strategy.
- Teams often inspect this once and assume future subsets are unchanged.
- Keep summary outputs alongside calibration logs for review.

# Citations

- smirnovRevisitingRadioInterferometer2011

