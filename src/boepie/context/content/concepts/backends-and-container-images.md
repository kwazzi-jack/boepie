---
type: Concept
title: Backends and container images
description: How backend selection and image execution affect portability, reproducibility, and skip behavior.
tags: [stimela, backends, containers, reproducibility]
managed_by: boepie
---

# Backends and container images

Stimela runs cabs through a backend that defines where and how execution happens. Container-backed execution gives repeatable environments across laptops, clusters, and cloud workers.

This matters because recipe behavior can differ between local and remote filesystems. A workflow that skips work based on local file freshness may not behave the same when the backend filesystem is remote.

```yaml
my_recipe:
  backend:
    select: default
  steps:
    run_tool:
      cab: some.cab
      params: {}
```

If used wrongly, the recipe runs but portability fails: paths resolve differently across hosts, image availability differs by site, or skip semantics are ignored on remote filesystems.

# Citations

- smirnovAfricanusIVStimela22025
- perkinsAfricanusScalableDistributed2025

