---
type: Concept
title: Aliases and parameter passing
description: How aliases expose step parameters at recipe level, and where alias conflicts break prevalidation.
tags: [stimela, aliases, parameter-passing, recipes]
managed: boepie
---

# Aliases and parameter passing

Aliases map recipe-level names onto step parameters. They make a recipe reusable by exposing only the controls users should tune, while hiding low-level wiring.

This matters when building playbooks into reusable templates. A clean alias surface keeps agent prompts short and reduces accidental edits to internal step plumbing.

```yaml
my_recipe:
  aliases:
    image_prefix: [image_step.output_name]
  steps:
    image_step:
      cab: some.imager
      params: {}
```

If used wrongly, prevalidation fails on unknown targets, mixed input and output aliasing, or conflicting alias definitions that do not agree on schema.

# Citations

- smirnovAfricanusIVStimela22025

