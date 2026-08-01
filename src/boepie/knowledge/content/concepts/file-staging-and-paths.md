---
type: Concept
title: File staging and paths
description: How recipe-relative and runtime paths are formed, and how path mistakes surface as missing outputs.
tags: [stimela, paths, staging, outputs]
managed: boepie
---

# File staging and paths

Stimela recipes commonly build path trees from recipe state and pass those paths through steps. Good path staging keeps outputs predictable, lets follow-up cabs consume products directly, and supports reproducible reruns.

This matters when many tools touch the same measurement set and image products. Explicit staged locations avoid accidental overwrite, make debugging easier, and keep quality-control products discoverable.

```yaml
my_recipe:
  assign:
    workdir: products
  steps:
    process:
      cab: some.cab
      params:
        output: =recipe.workdir + "/result"
```

If used wrongly, later steps fail with missing files or stale products are reused. The common failure pattern is inconsistent path construction across steps in the same workflow.

# Citations

- smirnovAfricanusIVStimela22025

