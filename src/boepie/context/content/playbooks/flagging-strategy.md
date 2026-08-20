---
type: Playbook
title: Flagging strategy
description: Practical sequence for conservative-to-aggressive flagging with reversible checkpoints.
tags: [playbook, flagging, rfi, quality-control]
managed_by: boepie
---

# Flagging strategy

Task: remove corrupted visibility samples while preserving as much scientifically useful data as possible.

Cab sequence:

1. `shadems` - inspect visibility distributions to identify whether corruption is localized or broad.
2. `casa.flagman` - create a recoverable checkpoint before applying new flag operations.
3. `tricolour` - run robust statistical RFI flagging as the primary automated pass.
4. `casa.flagdata` - apply targeted manual or rule-based flagging for leftovers not captured in automated pass.
5. `taql.update` - perform narrow surgical edits for corner cases best expressed as table logic.
6. `casa.flagman` - checkpoint the cleaned state for later calibration branches.

Decision points:

- If contamination is narrow-band or short in time, prefer targeted operations and avoid broad masks.
- If broad structures remain after automated flagging, escalate with conservative manual strategies first, then stronger actions.
- If a new flag pass removes large contiguous coverage and worsens calibration stability, roll back to the prior checkpoint and re-plan.

# Citations

- mitchellRealTimeCalibrationMurchison2008
- tasseNonlinearKalmanFilters2014

