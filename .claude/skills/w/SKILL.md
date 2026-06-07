---
name: w
description: Load the workspace and context as preparation to work on the "90ci" project. Use when user asks to "load the workspace", "work on this project", "work on 90ci" or "resume working".
---

# W: working on Tre
Follow the "Instructions" to get started, then use the guidelines as you continue to work.

## Instructions
- Read the `DOCS/INDEX.md` for context on previous work.
- Check if `DOCS/CANDIDATE_RULES.md` has any entries. The file should be empty, so if it's not warn the user that there's candidate rules to be processed.

## Guidelines
- IF a new doc is written in `DOCS/`, THEN add an entry for it in `DOCS/INDEX.md`.
- IF something important is learned (e.g.: a tool error is found and overcome), THEN capture it as an entry in `DOCS/CANDIDATE_RULES.md` and inform the user.
