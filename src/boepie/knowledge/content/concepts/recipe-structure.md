---
type: Concept
title: Stimela recipe structure
description: How recipe, steps, and cabs fit together, and what fails when that structure is invalid.
tags: [stimela, recipe-language, steps, cabs, structure]
managed: boepie
---

# Stimela recipe structure

A Stimela recipe is a named workflow with declared inputs and ordered steps. Each step runs one cab or one nested recipe, so the control flow is readable from top to bottom.

This matters because most agent mistakes in early drafting are structural rather than scientific. If the recipe shape is valid, iteration is fast: you can swap cabs, add branches, and preserve a stable execution skeleton.

```yaml
my_recipe:
  inputs: {}
  steps:
    first:
      cab: some.cab
      params: {}
    second:
      recipe: another_recipe
      params: {}
```

If used wrongly, validation fails before work starts. Typical failures are a step that declares both `cab` and `recipe`, a step that declares neither, or a step label referenced elsewhere that does not exist.

# Citations

- smirnovAfricanusIVStimela22025
