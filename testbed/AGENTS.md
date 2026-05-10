# AGENTS.md

This file defines repository-wide working rules for coding agents in this
workspace.

## Scope

These rules apply to the entire repository unless a deeper `AGENTS.md` overrides
them for a subdirectory.

## Global Rules

1. Language

All user-facing responses should be in Chinese by default, unless the user
explicitly asks for another language.

2. Documentation Sync

Any task that changes code must also update the relevant documentation in the
same task. Update the closest source of truth for the change, such as:

- `README.md`
- files under `docs/`
- protocol or integration specs
- config examples
- docstrings or other developer-facing documentation when appropriate

If a code change is purely internal and no documentation update is needed, the
agent must explicitly justify that decision in the final response instead of
silently skipping documentation updates.
