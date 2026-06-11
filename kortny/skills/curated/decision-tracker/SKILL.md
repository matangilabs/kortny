---
name: decision-tracker
description: Use when tracking open decisions across days, threads, or check-ins — every decision carries an owner, options, and deadline; unknowns become explicit placeholders; closed decisions are acknowledged once and dropped.
metadata:
  version: 1.0.0
  display_name: Decision Tracker
  tags: decisions, tracking, owner, deadline, follow-up
---

## Goal

Keep every open decision visible until it closes — and gone the moment it does.

## Steps

1. Record each open decision with three fields — owner, options, deadline: "2. Venue: office vs. rooftop — Dana decides by Friday."
2. Hold known-unknowns as explicit placeholders, never silence: "[vendor shortlist — waiting on procurement]". A placeholder is a promise to fill it; chase the owner when it ages more than a day or two.
3. Restate open decisions verbatim day-to-day until closed — same wording, same numbering. Stable text is what makes the list scannable.
4. When a decision closes, acknowledge it exactly once ("Venue's settled — rooftop it is") and drop it from every future list. Never renumber the survivors mid-week without saying so.
5. For decisions that should outlive the thread — named owners, standing policies, recurring deadlines — propose a durable fact with remember_fact so the workspace remembers after the thread scrolls away.

## Rules

- A decision without an owner isn't tracked, it's lost — flag it "(no owner)" until someone claims it.
- Never re-litigate a closed decision; if new information reopens one, say that explicitly and re-add it as a new numbered item.
- Placeholders name what's missing and who it's waiting on, in brackets, every time.
