# Candidate rules

Learnings staged for promotion into `.claude/CLAUDE.md`. Each follows the `ALWAYS/NEVER X BECAUSE Y` form.

### Front-end (Stage 6)
- ALWAYS assert tooltip/hint reachability through the accessibility tree (`toHaveAccessibleDescription`, `toHaveAccessibleName`), never via `getByText` on a CSS-hidden element BECAUSE jsdom applies no CSS, so the always-present node is found regardless of `:hover`/`:focus` visibility and the test passes vacuously.
- ALWAYS import a component's exported copy constant into its test and assert against it, never re-declare a local literal of the same string BECAUSE a local copy still matches the old text after the component's constant changes, so the "copy review" silently stops protecting against drift.
- ALWAYS expose descriptive helper text (a tooltip, an icon's meaning) via `aria-describedby` on the control it annotates, never as bare child text of an element that already has an `aria-label` BECAUSE the accessible-name computation lets the `aria-label` override the child text, so screen-reader users never receive the description even when it is keyboard-focusable and visible to sighted users.
