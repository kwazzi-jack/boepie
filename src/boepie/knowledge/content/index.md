# Stimela knowledge base

Selective map of the curated knowledge bundle. Read this file first, then
jump straight to the file that matches your question - do not browse the
whole tree.

## Escalation ladder

| Question shape | Where to look |
|-----------------|----------------|
| "What are this cab's parameters / defaults / choices?" | `get_cab_schema`, `get_cab_docs`, `get_cab_params` (live, authoritative - never this bundle) |
| "How do I structure this pipeline / task?" | `playbooks/` |
| "What does this recipe-language feature mean, how do I write it?" | `concepts/` |
| "What does this specific cab do, when do I reach for it, what are its gotchas?" | `cabs/` |
| "Why does this algorithm behave this way?" | `literature/` stub, then `read_literature` via its citekey |

## Layout

- `concepts/` - recipe language, substitution, backends, loops
- `playbooks/` - task-to-cab strategies (imaging, selfcal, flagging)
- `cabs/` - what/when/gotchas per cab; no parameter tables, cites papers
- `literature/` - one stub per citekey: abstract + `read_literature` handle
- `log.md` - update history, newest first
- `manifest.json` - bundle_version, cultcargo_version, boepie_version, generated_at

## Rule

Parameter values, defaults, and choices are never stored here: they drift
with the installed cult-cargo version and go stale silently. Always get them
live through the MCP cab tools (L1), and treat this bundle (L2) as strategy
and concepts only.
