---
name: stage-dev
description: Coordinates agents to implement a stage of the implementation plan of `90ci` PDF. Use when user asks to "implement next stage", "implement stage" or "work on implementation".
---

# Stage dev

## Overview

Coordinates gathering info on the feature to be developed, work out the first implementation pass, a follow-up code review, and subsequent refactor/improvement pass to the code based on the reviewers feedback. This skill is meant to launch and coordinate the work of multiple concurrent agents.

## Operational model

The workflow is carried out in four steps:

1. Preparatory work, by the main agent
2. Feature development, carried out by an agent team of developers
3. Code review, performed by a set of subagents
4. Refactor and fixes, done by the same agent steam as step 2.

### The actors

1. The agent loading this skill becomes the main agent
  - prepares the repo
  - gathers the requirements
  - launches agent team and subagents

2. The "developers" agent team
  - two teammates responsible of front-end and library/core development
  - developer teammates _are inter-dependent_ and require careful coordination
  - runs first pass for development, then a second pass after the reviewers produce feedback

3. reviewer subagents
  - launched by the main agent
  - main agent gathers their feedback into a report to pass back to developer team

## Instructions

### Step 1: gather the requirements

Loads the stage to be implemented from `DOCS/IMPLEMENTATION_PLAN.md` but may be elsewhere referenced by user. If the use-case/details are not found or is not clear _stop immediately_ and inform the user that further definition is needed.

### Step 2: prepare the dev environment

Use a sub-agent to verify the repo is in a "ready" state. The subagent's tasks are:
- Verify if `docker` is running (it will be needed for the E2E tests).
  - IMPORTANT: If subagent finds out that docker is NOT running, stop and request user intervention.
- Validate that there's nothing staged and no lingering changes in the working tree.
- Switch to the main branch, `master`, and create a new feature branch with the name schema `stage/XXX` where "XXX" is the stage number (example "1").
- Run the entire test suite (including E2E tests) to verify it's passing.
  - IMPORTANT: If subagent finds broken tests, stop and request user intervention.

### Step 3: delegate implementation

Create an agent team to handle the implementation: one teammate using `dev-rust` agent type for the core components (lib, WASM wrapper, etc.), and another using the `dev-front-end` agent type for the web UI side. The team must be informed that the entire test suite must be passing before the implementation can be called "done". This includes the E2E tests. When both teammates have effectively finished their job the team can exit (a new agents team will be created later on for the "second pass").

#### Step 3.1: update the map

Update `DOCS/MAP.md` with the developers work. The updated doc will be used by the reviewers.

#### Step 3.2: commit first pass

`git commit` in the feature branch. Using the "Git commit message guidelines" section below to write the commit message.

### Step 4: delegate code review

#### Step 4.1: launch the specialized sub-agents
- Spin off the following sub-agents to review the changes independently:
  - `reviewer-front-end` for code quality recommendations about changes in `web/`
  - `reviewer-rust`  for quality and correctness of the core code and WASM interface.
- Share `DOCS/MAP.md` with each agent.
- When generating their prompts, insist that each should identify at least one issue OR explicitly justify clearance with evidence.

#### Step 4.2: Aggregate the reports and produce a deliverable

- Collect output of reviewer subagents
- Identify commonalities and overlap, and resolve conflicts and duplication.
  - If resolution is non-obvious, escalate to the user.
- Identify and flag when a recommended fix or improvement solves multiple issues raised by different subagents.
- Decide if the changes effectively address the stage and requirements:
  - Do the code changes address the original intention fully?
  - Do the changes do things that are not part of the requirements?
- Generate the final, aggregated report
  - Add an overall recommendation of whether the changes reviewed are healthy or not based on the presence of any critical bugs or issues.
  - Include summary of code changes purpose and the evaluation resulting from step 5.
  - Add the minimal possible text to the aggregated report.

### Step 5: second implementation pass

Create a new agent team to refactor the code according to the feedback from the reviewers. Much like in Step 3, the team has two members: one teammate using the `dev-rust` agent type for Rust components and another using the `dev-front-end` agent type for the web UI side. The team must be informed that the entire test suite must be passing before the implementation can be called "done". This includes the E2E tests. When both teammates have effectively finished their job the team can exit.

Share the aggregated review report with the developer agent team, instructing it to "make a second pass" following this guidelines:
  - Requested changes should be addressed  minimal changes that effectively address the issue.
  - Suggested changes may be ignored, but only if the child agent has thoroughly analyzed the issue and found a strong argument against addressing it (example: "change contradicts DOCS/architecture/X.md", "change involves substantial work in entire app").
    - This is considered "push-back by the developer"
    - If ignored, the suggested change is not addressed, but _is reported as part of the deliverable_.
  - The entire test suite must be passing before the implementation can be called "done". This includes the E2E tests.

#### Step 5.1: commit first pass

`git commit` in the feature branch. Using the "Git commit message guidelines" section below to write the commit message.

### Step 6: Post-work and final report

Report on the status of `90ci` after the stage has been implemented (or failed to): Status of branches touched and pending decisions.
And perform the following tasks:
- Add learnings to `DOCS/CANDIDATE_RULES.MD` and let the user know
- Update `DOCS/MAP.md` 
- Finally, `git commit` the docs changes.

## Git commit message guidelines

These guidelines are to be used for committing the first and second pass of implementation.

- A short one-line summary followed by an empty line. Example "first pass: stage 5"
- After that comes the description:
  - Short logical summary of what the changes accomplish
  - Empty line
  - Full list of files edited
