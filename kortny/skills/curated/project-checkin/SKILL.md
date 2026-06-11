---
name: project-checkin
description: Use when posting a recurring project check-in, morning status update, or daily standup-style message for a project or channel — gather schedules, thread context, and open decisions first, then post a tight check-in with deadline countdowns.
metadata:
  version: 1.0.0
  display_name: Project Check-in
  tags: check-in, status, project, recurring, standup
---

## Goal

Run a recurring check-in the team actually reads: current state, open decisions, live deadlines — nothing else.

## Steps

1. Gather state before writing a word: active schedules, the channel/thread since the last check-in, workspace facts and graph commitments, and the open-decision list from the previous check-in.
2. Lead with a one-line project state: "On track for the 24th" or "At risk — decision 2 is the blocker."
3. Enumerate open decisions as a SHORT numbered list. Restate each one verbatim from the prior check-in until it closes — same wording, same number, so nobody has to diff yesterday's message.
4. Put a countdown on every live deadline: "7 days to June 18."
5. If a target is at risk, offer the slip explicitly and make accepting it cheap: "totally fine to slip to Thursday — say the word."
6. Close with exactly one concrete turnaround promise: "Revised timeline posted here by 2pm."

## Rules

- Never pad. If nothing changed, the whole check-in is one line: "No movement since yesterday — decisions 1 and 2 still open."
- New information goes at the top, standing items below it.
- Decisions and deadlines come from gathered state, never from memory of what "should" be open.
- One turnaround promise, not a list of intentions — and keep it.

See `references/example.md` for a complete example check-in.
