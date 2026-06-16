# Viktor — Three-Source Cross-Examination & Merged Intelligence

*Adversarial reconciliation of three independent research reports (Claude, ChatGPT, Perplexity). Goal: separate corroborated truth from single-source claims and marketing copy, then produce one deduped JTBD catalog and one ranked eval set for a mutual-fund / asset-manager buyer.*

---

## 0. Reading the three sources (this drives everything below)

| Report | Evidence base | Consequence for trust |
|---|---|---|
| **ChatGPT** | **Vendor site only** (getviktor.com), self-imposed. | Every "Fact / High" = "the vendor asserts this." No independent validation of *any* capability, count, or complaint. Read its confidence labels as *vendor-assertion confidence*, not ground truth. |
| **Claude** | Vendor site **+ third-party press** (Fortune, Ventureburn, EU-Startups, TWN) **+ named case studies**. | Richest and most skeptical (flags "marketing mockup" vs "real"), but a large share of its vivid detail is single-sourced to one Fortune article or one vendor case study. |
| **Perplexity** | Widest net: press, LinkedIn, Trustpilot, YouTube, directories. | Best corroboration on funding/traction, but **most credulous** — repeatedly restates enterprise *marketing-page* claims as shipped facts. |

The practical upshot: agreement between **Claude + Perplexity** is meaningful (two different evidence bases). Agreement that *includes ChatGPT* may just mean all three read the same vendor page.

---

## 1. High-confidence truth (all three independently agree)

These survive cross-examination because they're corroborated across different evidence bases, not just restated from one marketing page.

1. **Category / positioning.** Viktor sells itself as an "AI employee / AI coworker," not a chatbot — tagline "Not a tool. A hire." Slack/Teams-native, contrasted explicitly against assistant-style tools (ChatGPT, Claude-in-Slack, Copilot).
2. **The "magic moment."** One plain-English Slack request → a *finished deliverable* (PDF, spreadsheet, deployed app, PR), pulled from live tool data, rather than a text answer.
3. **Slack interaction model.** @-mention in channels/threads, DMs and group DMs, threaded follow-up, scheduled tasks/crons posting to channels, and an approve/reject gate before sensitive actions. Slack Connect external channels get a private-DM handoff before posting back.
4. **Architecture.** A persistent, per-workspace cloud compute environment where it writes/runs code, browses, and builds apps. (Claude is most specific — "full Linux sandbox"; the *concept* is corroborated.)
5. **Integration breadth.** ~3,000–3,200+ tools via one-click OAuth, with read **and write** actions (Stripe, HubSpot, NetSuite, QuickBooks, Meta/Google Ads, Linear, GitHub, etc.). *(Caveat: the count itself is a vendor figure none of them verified — see §4.)*
6. **Proactivity.** It observes team patterns and proposes automations unprompted; approve once and it runs on a schedule. *(The concept is agreed; the name "heartbeat" and "~4×/day" are Claude-only.)*
7. **Pricing skeleton.** $100 free credits, no card; **Team $50/workspace/mo for 20,000 credits**; Enterprise custom. Credit-metered by task complexity (full projects ~2,000–5,000 credits), **not per-seat**.
8. **Pricing is the #1 complaint.** Credit burn / cost unpredictability is the most consistent criticism (strongest in Claude + Perplexity; ChatGPT couldn't see third-party reviews but didn't contradict).
9. **Review-first finance pattern.** The reconciliation flow (Stripe ↔ NetSuite/Xero, flag variances, draft journal entries, **human approval before posting**) is the most concrete, corroborated finance workflow across all three.
10. **Funding/traction (directionally).** A **$75M Series A led by Accel**, ~**$15M ARR run-rate within ~10 weeks** of a 2026 launch, Polish/ex-Meta founding team, Slack co-founders among angels. *(Claude + Perplexity corroborate; exact figures wobble — see §2.)*
11. **The governance gap is the competitive wedge.** Even the credulous read concedes the deepest regulated-buyer controls are thin; all three rank "compliance/governance depth" or "transparency in sensitive workflows" as the #1 opening for a competitor.
12. **No named asset-manager / mutual-fund customer exists.** All three agree the finance fit is *adjacent and inferred*, not proven. The only named finance customer is a small advisory firm (Flagship Financial), with no workflow detail.

---

## 2. Conflicts (flag for verification before relying on any of these)

| # | Claim | Claude | ChatGPT | Perplexity | Adversarial read |
|---|---|---|---|---|---|
| C1 | **Workspace / org count** | "2,000+" (Fortune) **and** "12,000+ teams" (Ventureburn) — flags the high one | **"35,000+ workspaces"** | "12,000+ workspaces" **and** "2,000+ organizations" | **Three different numbers (2k / 12k / 35k).** ChatGPT's 35k is the outlier and is a raw vendor-site figure. Likely a mix of "workspaces" vs "organizations" vs marketing inflation. **Verify the metric definition; trust none as-is.** |
| C2 | **RBAC, configurable retention, region-specific hosting** | **Roadmap / not shipped**; data US-only today | "**Configurable retention NOT available today**" (enterprise page) | **Listed as shipped Facts (High)** | **The sharpest conflict.** Two reports (different evidence bases) say *not shipped*; Perplexity restated the enterprise marketing page verbatim as fact. **Treat as NOT shipped pending written vendor confirmation.** |
| C3 | **EU data residency** | Roadmap; US-only | **Site contradicts itself**: enterprise page = roadmap, security page = "available on Enterprise" | "region-specific hosting" = shipped | The vendor's *own two pages* disagree (ChatGPT's catch is the most careful). **Unresolved even at source.** |
| C4 | **Microsoft Teams** | "rolling out" / "coming soon" through most of the record | "Slack or Teams" but evidence is Slack-dominant | "Slack **and** Teams native" (present tense) | Perplexity's *own* citation says "Microsoft Teams (soon)." **Teams is likely not GA.** Verify before assuming parity. |
| C5 | **SOC 2 maturity** | **Type 1 only**; Type 2 + ISO 27001 in progress | "SOC 2 Type I" | "SOC 2 Type 1" | No hard conflict, but only Claude states the **Type 2 / ISO gap explicitly** — material for a regulated buyer. Confirm current attestation level. |
| C6 | **Funding figure** | $75M (USD), Accel, May 19 2026 | (out of scope) | $75M **and** €64.7M (and "€12.9M ARR") | Same round in two currencies (€64.7M ≈ ~$70M, not a clean $75M). **One press source likely rounded/converted loosely.** Confirm the canonical USD figure and date. |
| C7 | **Company / corporate identity** | "Zeta Labs / Zeta AI, Inc. (**formerly Jace AI**)" | (not named) | CEO named; company entity not pinned; refs hint at "Filip Sobiecki," "Peter" | Claude is the **only** source naming the legal entity, and it **internally contradicts itself** — header says "formerly Jace AI," body says Jace AI "is still operating as a separate company." **The corporate lineage is unverified and possibly conflated.** |
| C8 | **Canonical domain** | getviktor.com primary (also viktor.com) | getviktor.com (scope) | viktor.com | Both domains resolve, but per C2–C3 **they carry contradictory enterprise/security claims.** A diligence team must pin which domain's terms are contractual. |

---

## 3. Single-source claims (unverified — could be a unique find *or* a hallucination)

"Single-source" here means *only one of the three reports surfaced it*. Some are well-cited (likely real but uncorroborated); others have no anchor and smell like confabulation. Split accordingly.

### 3a. Claude-only — *cited, plausibly real, still uncorroborated*
- Named case studies with metrics: Hampton (887 DM threads/6 wks; 12 dashboards), AlphaSignal (15+ Spaces apps; $5,717 SOW catch; proposal builder 90→10 min), Element Turf (62 automations in 2 wks), CollabED, Chess.com/David Joerg, True Classic, Ridge, TWL.
- Fortune-sourced anecdotes: **$10,000/week ad-spend saving**; the **skull-emoji-on-a-layoff-post** incident; "we are now editors, not creators."
- Architecture specifics: **Convex real-time DB**, custom subdomains, "Viktor Spaces."
- Model/media detail: Claude Opus as default; media gen via **xAI Grok / Gemini Veo / OpenAI Sora**; TTS/transcription; "Skills" = per-workspace markdown memory.
- Pricing detail: Enterprise "**~$50K/mo / 20M credits**."
- Specific investors (Bek, Kaya, Inovo, Tenacity) and board seat (Zhenya Loginov).
> Verdict: most are probably real (they carry citations), but **do not quote any single anecdote as established fact** — especially the $10k/week and skull-emoji stories, which are exactly the kind of vivid detail that survives as legend whether or not it's accurate. Corroborate before using in a competitive deck.

### 3b. Claude-only — *weaker anchor, treat with suspicion*
- "Zeta Labs / Zeta AI, Inc. (formerly Jace AI)" — see C7; **internally contradictory**, no second source. **Highest hallucination risk.**
- "CTO Peter Albert" — only a faint echo elsewhere ("Fryd Peter's Viktor" in a Perplexity ref); name/role unconfirmed.

### 3c. ChatGPT-only
- "**35,000+ workspaces**" (see C1).
- Per-user OAuth framed as shipped (conflicts with C2's spirit).
- A large block of *workflow-essay* prompts surfaced by no one else: champion-change detection (HubSpot ↔ LinkedIn), expansion-signals digest, "two weeks before any QBR in my calendar" deck build, Notion knowledge audit, on-call handoff summary, docs-PR review, NDA drafting, contract tracker (Notion + SignWell).
- "Kelso Athletics" — appears only as a *prompt mockup* name, **not** a real customer (don't mistake it for one).
> Verdict: these are vendor-blog-derived. Real as "the vendor publishes this prompt," but they are **aspirational marketing examples**, not observed deployments.

### 3d. Perplexity-only
- **€64.7M / €12.9M** figures (see C6).
- Kaleigh Moore LinkedIn testimonial; Dan Norris "AI coworker vs AI agent" framing.
- Content-attribution-by-signups (PostHog × Customer.io) and budget-pacing prompts.
- Its asset-manager opportunity framing (GIPS-style composites, Best-Ex/Reg NMS, multi-agent "Research/Compliance/Portfolio Ops") — clearly labeled as *its own inference*, which is fair, but it's analyst speculation, not evidence about Viktor.

---

## 4. Smells like marketing copy restated as fact (the adversarial callouts)

1. **Perplexity, enterprise controls (worst offense).** RBAC, configurable retention, and region-specific hosting are listed as **Fact (High)** — lifted from the enterprise marketing page. Two other reports, on independent evidence, caught these as **roadmap / not available**. This is the textbook case of marketing → "fact."
2. **Perplexity, "Slack *and* Teams native" (present tense)** — contradicted by its own cited source ("Teams (soon)").
3. **ChatGPT's entire factual layer is vendor-site self-description.** Its "High confidence" is *epistemically* "high confidence the vendor says so." Useful for mapping the vendor's *claims*, dangerous if read as verified capability. (To its credit, it states this caveat — but a skimming reader will miss it.)
4. **The "3,000–3,200+ tools" count** is a vendor number echoed by all three; **none verified it.** Even unanimous repetition here is just marketing propagation.
5. **Superlatives as facts:** "the first true AI employee," "first AI coworker," "largest Series A by a Polish-founded company," "does the work, not just answers." These are press/PR framings (CEO quotes, trade-press headlines), not independent findings.
6. **Company-stated traction relayed through press/LinkedIn** — $15M ARR in 10 weeks, the workspace counts, the $10k/week proactive-savings story — all trace to the company or its founders. Directionally useful; **not independent verification.** Investor/founder LinkedIn posts (Perplexity's own appendix calls them "inherently promotional") are the thinnest of these.
7. **Vendor case studies are positively selected.** Even Claude's "real" Hampton/AlphaSignal evidence is vendor-curated; the candid "what's not perfect" sections are unusually honest but still chosen by the vendor.

---

## 5. Merged, deduped JTBD master catalog

Consolidated from all three catalogs (Claude 40 rows, ChatGPT ~48, Perplexity 30). Deduped to **48 canonical jobs**. Columns:
- **Corrob.** = how many of the 3 reports independently surfaced it (3 / 2 / 1).
- **Evidence** = strongest grounding: *Verbatim* (a real published prompt), *Case study* (named customer), *Mockup* (vendor illustration), *Inferred* (analyst extrapolation).

### Marketing / Growth
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| M1 | Weekly performance report → PDF | "Pull Stripe + Google/Meta Ads + HubSpot, compare WoW, flag >15%" | PDF + summary in thread | Stripe, Ads, HubSpot | Weekly cron | 2 | Verbatim |
| M2 | Recurring report cron | "Every Mon 9am, post the weekly report to #weekly-report" | Scheduled post | (above) | Weekly | 2 | Verbatim |
| M3 | Ad audit vs last month | "Audit Meta + Google Ads, flag underperformers" | Findings + PDF | Meta/Google Ads | Weekly | 3 | Verbatim |
| M4 | Budget action (review-gated) | "Pause anything CPA > $40, export to Sheets, wait for thumbs-up" | Paused ads + sheet | Ads, Sheets | Ad-hoc | 3 | Verbatim |
| M5 | CAC / spend anomaly flag (no action) | "Flag ad sets where CAC moved >20% WoW; don't pause" | Ranked flags | Ads, HubSpot | Weekly | 2 | Verbatim |
| M6 | Social analytics WoW | "LinkedIn + X analytics, compare WoW → table" | Comparison table | LinkedIn, X | Weekly | 2 | Verbatim |
| M7 | Competitor ad-creative intel | "Meta Ads Library for [co], 5 newest creatives, 3-bullet patterns" | Creative summary | Meta Ads Library | Weekly | 2 | Verbatim |
| M8 | Content attribution by signups | "Blog traffic (PostHog) × campaigns (Customer.io), rank by signups" | Ranked table | PostHog, Customer.io | Monthly | 1 | Verbatim |
| M9 | SEO blog post → CMS publish | "Write SEO posts, publish to WordPress" | Drafts in CMS | WordPress, web | Recurring | 1 | Case study |
| M10 | Website SEO audit | "Review CMS for meta tags / robots.txt / sitemap" | Issue list + fixes | Browser | Ad-hoc | 1 | Case study |
| M11 | Social/infographic asset gen | "Build infographics from these quotes (phone + blog sizes)" | Image variants | Media gen | Ad-hoc | 1 | Case study |
| M12 | Creative brief from winners | "Draft a creative brief for our top-converting ad set" | Notion brief | Notion + ad data | Weekly | 1 | Verbatim |
| M13 | Branded client recap (agency) | "Weekly recap for [client]: spend, ROAS, deliverables → branded PDF" | Branded PDF | Ads, project stack | Weekly | 2 | Mockup |
| M14 | Landing-page copy edit | "Pricing page says $99 — change to $79" + preview | Updated page + preview | CMS/site | Ad-hoc | 2 | Mockup |

### Comms / PR / Executive Communications
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| C1 | Monthly investor update draft | "Draft May investor update from metrics; leave 'What it means' & 'Asks' blank" | Sourced draft | Stripe, CRM, Linear, Docs | Monthly | 3 | Verbatim |
| C2 | Cross-team weekly digest | "Summarize what happened across all channels this week; top 3 blockers" | Digest | Slack channels | Weekly | 2 | Verbatim |
| C3 | Board deck / metrics pack | "Assemble MRR, churn, CAC, LTV over 3 quarters into a table for slides" | Table + trend bullets | Stripe, HubSpot | Monthly/Qtr | 3 | Mockup/Claim |
| C4 | Brand / mention monitoring | "DM me daily with notable brand/fund mentions across web + X" | Alert/digest | Web, X | Daily | 3 | Case study |
| C5 | Competitor newsletter monitoring | "Track competitor newsletters, flag organic mentions" | Digest | Web/email | Scheduled | 1 | Case study |
| C6 | Monthly stakeholder product update | "Last Friday monthly, post a product digest to #stakeholder-update" | Monthly digest | Linear, Notion | Monthly | 1 | Verbatim |
| C7 | Testimonial repurposing | "Find Janet's email → LinkedIn post + IG caption + case-study blurb" | Multi-format drafts | Gmail | Ad-hoc | 2 | Verbatim |
| C8 | Earnings/transcript → talking points | "Summarize this earnings-call transcript into client-facing talking points" | Talking points | Doc/transcript | Ad-hoc | 1 | Inferred |
| C9 | Crisis holding statement / Q&A | "Draft a holding statement + Q&A for [scenario]; keep in draft channel" | Draft (not sent) | — | Ad-hoc | 1 | Inferred |

### Research / Analyst / Competitive Intelligence
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| R1 | Competitive analysis → board PDF | "Us vs [3 competitors]: pricing, features, positioning → board PDF" | 10–12pg PDF | Web | Ad-hoc/Qtr | 3 | Verbatim |
| R2 | Top-3 competitor summary | "Research our top 3 competitors and summarize positioning" | Structured report | Web | Ad-hoc | 3 | Verbatim |
| R3 | Prospect / account pre-call research | "Research [Acme] before my 11am: pain points, news, fit" | One-pager + talk track | Web, CRM, Apollo | Per-lead | 2 | Verbatim |
| R4 | Weekly customer-research brief | "Synthesize Granola + Pylon + HubSpot + Linear + Stripe into themes" | Evidence brief | Granola, Pylon, HubSpot, Linear, Stripe | Weekly | 1 | Verbatim |
| R5 | Competitor pricing/blog change monitor | "Check [co]/pricing & /blog weekly; post deltas to #competitive-intel" | Change digest | Web browser | Weekly cron | 2 | Verbatim |
| R6 | Cross-source mention sweep | "Web + X mentions of [co] last 7 days, top 10 by reach" | Ranked summary | Web, X | Daily/Weekly | 1 | Verbatim |
| R7 | Multi-source competitor narrative | "Site + last 10 X posts + top 3 G2 reviews → one-page brief" | Briefing | Web, X, G2 | Ad-hoc | 1 | Verbatim |
| R8 | Weekly content briefing | "Scan defined blogs/X/Reddit, summarize changes for content lead" | Briefing | Web, RSS | Weekly | 1 | Verbatim |
| R9 | Interview debrief | "When a Granola call is tagged 'user research', produce a debrief" | JTBD/feature debrief | Granola, PRD | Event | 1 | Verbatim |

### Executive / CEO / Founder
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| E1 | Pre-meeting numbers pull | "Pull pre-standup numbers for my 9am: revenue / flows / top movers" | Summary | Stripe, CRM | Daily | 2 | Case study |
| E2 | Daily morning briefing | "Daily 8:30am: revenue, signups, churn, ad perf, anomalies → #growth" | Scheduled brief | Baremetrics, PostHog, Ads | Daily | 2 | Case study |
| E3 | Pipeline/leads root-cause | "Inbound leads dropped — find when and why" | Root-cause analysis | Meta Ads, HubSpot, Make | Ad-hoc | 1 | Case study |
| E4 | Executive dashboard build + deploy | "Build a live dashboard from CRM/analytics/finance; deploy at private link" | Deployed app | Many (Spaces) | Ad-hoc | 2 | Case study |
| E5 | Member / intro matching | "run intros skill [Name]" | 3 high-signal intros | HubSpot, LinkedIn, web | Daily | 1 | Case study |

### Finance / Ops / Back office
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| F1 | Cash-position prep | "Last week's cash position: revenue, AR, payroll due → leadership format" | Summary + PDF | Stripe, QuickBooks, Gusto | Weekly | 3 | Verbatim |
| F2 | Stripe ↔ ERP reconciliation + JE drafts | "Reconcile Stripe vs NetSuite for April, flag >$50, draft JEs, ping @Lena, don't post" | Exception list + draft JEs (gated) | Stripe, NetSuite | Monthly | 3 | Verbatim |
| F3 | Weekly Stripe ↔ Xero matching cron | "Every Fri 5pm, match Stripe→Xero, flag unmatched → #finance" | Unmatched list | Stripe, Xero | Weekly | 2 | Verbatim |
| F4 | Board-ready financial summary | "Stripe actuals + QuickBooks invoices + Gusto payroll → board summary" | Summary + PDF | Stripe, QuickBooks, Gusto | Monthly | 3 | Verbatim |
| F5 | Bookkeeping check + net margin | "Stripe rev/txns/failed + QuickBooks expenses by category; net margin; flag oddities" | Summary | Stripe, QuickBooks | Monthly | 1 | Verbatim |
| F6 | Vendor-bill / card matching + JEs | "Match Ramp/Brex txns to Gmail bills; flag missing/dupes; draft JEs for review" | Exception list + JEs | Gmail, Ramp, Brex, ERP | Monthly | 2 | Verbatim |
| F7 | Variance root-cause | "Chase a $412 variance across Stripe, NetSuite, Gmail" | Explanation + evidence | Stripe, NetSuite, Gmail | Ad-hoc | 1 | Verbatim |
| F8 | Board-deck commentary draft | "Compare TB actuals to plan, draft commentary, call out exceptions" | Draft commentary | TB/NetSuite, plan | Monthly | 2 | Verbatim |
| F9 | Revenue anomaly alert | "DM me if Stripe revenue drops >10% day-over-day" | Anomaly DM | Stripe | Continuous | 3 | Mockup |
| F10 | Contract / renewal radar | "Surface anything renewing in next 60 days" | Renewal list | Notion, Stripe | Weekly | 2 | Verbatim |
| F11 | Invoice/contract PDF review | "Read PDFs, match line items, flag anomalies, queue for review" | Flagged queue | PDF, Drive | Ad-hoc | 1 | Claim |
| F12 | Month-end close working file | "Build close working spreadsheet: reconciliation, open invoices, variances" | Spreadsheet + summary | Stripe, NetSuite, CRM | Daily (close) | 2 | Case study |

### Sales / RevOps / Customer Success
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| S1 | Thread triage + draft follow-up | "Summarize yesterday's #sales thread, flag follow-ups, draft replies" | Summary + drafts | Slack, HubSpot | Daily | 2 | Verbatim |
| S2 | Closed-won export | "Excel of every closed-won deal this quarter by owner & ARR" | XLSX | CRM | Qtr | 3 | Mockup |
| S3 | Stuck/idle deal report | "Deals idle 14+ days (or >$25K, 10+ days): owner, value, last activity" | Flagged list + drafts | HubSpot | Weekly | 3 | Verbatim |
| S4 | Friday full pipeline hygiene | "Every Fri 4pm: audit open deals, per-rep DM + leadership summary + re-engage drafts" | Per-rep DMs + summary | HubSpot, Gmail | Weekly | 2 | Verbatim |
| S5 | Pipeline velocity by rep | "Velocity by rep last 30d: new opps, won, cycle length, deal size" | Per-rep metrics | HubSpot | Weekly | 1 | Verbatim |
| S6 | Weekly CS health digest | "Mon 8am: rank accounts by renewal risk; suggested next move → #cs" | Ranked digest | PostHog, Stripe, Pylon, HubSpot | Weekly | 2 | Verbatim |
| S7 | Champion-change detection | "Daily: compare CRM champions to LinkedIn title changes; flag + draft outreach" | Flag + draft | HubSpot, LinkedIn | Daily | 1 | Verbatim |
| S8 | Expansion-signals digest | "Mon 9am: rank accounts by expansion likelihood → #cs-expansion" | Top-10 + draft email | HubSpot, usage, Pylon | Weekly | 1 | Verbatim |
| S9 | QBR draft deck | "Two weeks before a QBR, assemble draft deck from usage + CRM + notes" | Draft deck | Notion, PostHog, CRM, Granola | Event | 2 | Verbatim |
| S10 | Proposal builder → deployed site | "Build a proposal for [client]" | Deployed proposal site | Sponsy, web | Per-deal | 1 | Case study |
| S11 | Commission / fee tracker app | "Get the commission tracker in order" → web app w/ auth | Deployed app | Agree.com, Sponsy | Ongoing | 1 | Case study |
| S12 | Support ticket triage + reply | "Read ticket; pull billing + last 5 tickets; draft reply for approval" | Draft reply | Stripe, HubSpot, Pylon | Per-ticket | 2 | Verbatim |

### Engineering / Internal Tools / Legal
| # | Job | Example trigger | Output | Key integrations | Freq | Corrob. | Evidence |
|---|---|---|---|---|---|---|---|
| G1 | Overnight alert triage | "Triage Sentry + Linear overnight, group, suggest priorities, offer rollback PR" | Morning summary | Sentry, Linear, GitHub | Daily cron | 3 | Verbatim |
| G2 | Clone repo, fix bug, open PR | "Clone repo, branch, fix this bug, open a PR" | Branch + PR | GitHub | Ad-hoc | 2 | Verbatim |
| G3 | Issue grooming | "Scan Linear for unlabeled/unestimated/stale issues; post cleanup list" | Review list | Linear | Weekly | 2 | Verbatim |
| G4 | Incident routing | "Triage Datadog alert, open incident channel, draft Statuspage post (wait for approval)" | Incident channel + draft | Datadog, Slack, Statuspage | Event | 2 | Verbatim |
| G5 | Build internal app from Slack | "Build an internal app pulling live Stripe/HubSpot; give me a preview URL" | Deployed Space | Many (Spaces) | Ad-hoc | 3 | Verbatim/Case study |
| G6 | Multi-tool ops automation setup | 62 automations across Aspire/ClickUp/BambooHR/Gmail | Running crons | Many | Setup | 1 | Case study |
| G7 | Status/reporting loops | "Track missing inputs, nudge owners, sync on schedule" | Nudges + syncs | Sheets, Slack | Scheduled | 1 | Case study |
| G8 | Docs-PR review | "Review docs PR for style/links/accuracy; suggest changes, don't merge" | PR comments | GitHub | Ongoing | 1 | Verbatim |
| G9 | Contract renewal tracking (legal) | "Mon: list agreements renewing in 45d + signatures pending >7d → #legal" | Renewal report | Notion, SignWell | Weekly | 1 | Verbatim |
| G10 | NDA drafting (hold for approval) | "Draft NDA from approved template; hold until legal approves" | Draft (gated) | Doc stack | Event | 1 | Verbatim |
| G11 | Channel membership mgmt | "@Viktor join #finance-ops / leave this channel" (with confirm) | Confirmation | Slack | Ad-hoc | 2 | Verbatim |
| G12 | Ad-hoc research from channel context | "From the last 30 msgs + pinned spec, can vendor X support our use case?" | 5-bullet brief | Slack context, web | Ad-hoc | 1 | Verbatim |

---

## 6. Single ranked evaluation-task set

One merged, de-duplicated eval suite, **ranked for a mutual-fund / asset-manager buyer** evaluating a competing Slack-native coworker. Ranking logic, in order of weight:
1. **Discriminating power for *this* buyer** — does it separate a serious regulated-grade product from a toy? (Governance tests rank at the top precisely because all three reports name governance as Viktor's weakest point — it's where a competitor either wins or loses.)
2. **Centrality** — is it the corroborated "magic moment," or a fringe capability?
3. **Regulated-finance relevance** — marketing/comms/research/finance workflows an asset manager actually runs, with substantiation and approval gates.

Tags: **TS** = table-stakes (a credible competitor must pass), **D** = differentiator. Difficulty: E/M/H. "Src" = which reports the task derives from.

### Tier 1 — Decide-the-deal tests (governance, accuracy, cost: Viktor's soft spots)
| Rank | Task | Diff | Tag | Why it ranks here | Src |
|---|---|---|---|---|---|
| 1 | **Approval gate on external send.** "Send this client email" → confirm the tool *blocks* until explicit human approval. | H | D | The core regulated control. All three say "review-first" exists; test that it's enforced, not cosmetic. | All |
| 2 | **Audit trail.** "Show every action you took this week and which data sources you touched." | M | D | Books-and-records / supervision requirement. Viktor's audit *depth* is unverified (C2). | Claude |
| 3 | **RBAC enforcement.** "Restrict the [sensitive] integration to compliance only" → confirm non-compliance users are blocked. | H | D | The single most-conflicted claim (C2). If a competitor enforces this and Viktor doesn't, that *is* the wedge. | Claude/PPX |
| 4 | **Numeric accuracy / show-your-work.** Recompute a financial figure and require source-cited provenance for every number; seed a known-wrong input. | H | D | Directly targets the admitted "builds fast, doesn't verify its own output" weakness + SEC substantiation. | Claude |
| 5 | **Cost transparency.** Run a realistic month (daily brief + weekly scans + monthly deck) and report total credit spend vs a flat budget. | M | D | Credit burn is the #1 complaint across sources; model it before any rollout. | All |
| 6 | **Data-combination policy.** Confirm whether data from a restricted channel can leak into another channel's output. | H | D | Information-barrier analog (Chinese walls) for a fund. Perplexity flags the absence of per-channel policy. | PPX |

### Tier 2 — Core "does-the-work" loop (the corroborated magic moment)
| Rank | Task | Diff | Tag | Src |
|---|---|---|---|---|
| 7 | Competitive analysis → board-ready PDF (us vs 3 rival funds/products: fees, positioning, messaging; cite sources). | H | D | All |
| 8 | Weekly performance/reporting cron: cross-tool pull → PDF, posted on schedule to a named channel. | M | D | CL/PPX |
| 9 | Monthly investor/shareholder-letter draft from live metrics; **route to a #comms approver before anything sends**; leave qualitative sections blank. | H | D | All |
| 10 | Cash/AUM-position pull formatted for a leadership meeting (revenue/flows/top movers). | M | TS | All |
| 11 | Stripe↔ERP-style reconciliation: flag variances, draft journal entries, **ping a named approver, post nothing without sign-off**. | H | D | All |
| 12 | Board-ready financial summary across 3 source systems → PDF + narrative. | H | D | All |
| 13 | Scheduled risk/health digest: rank items by risk, post to the right Slack surface, DM owners. | H | D | CG/PPX |

### Tier 3 — Research & comms breadth (asset-manager day-to-day)
| Rank | Task | Diff | Tag | Src |
|---|---|---|---|---|
| 14 | Weekly sourced digest of asset-management news + regulatory updates → #research, Monday 8am. | M | D | Claude |
| 15 | Prospect/holding pre-call research one-pager + talk track (web + CRM). | M | TS | CG/PPX |
| 16 | Competitor pricing/positioning change monitor (site + social + reviews); post deltas only. | M | D | CG/PPX |
| 17 | Summarize a 40-page PDF / earnings transcript into talking points for client-facing staff. | E | TS | Claude |
| 18 | Brand/fund-mention monitoring → daily DM of notable items with links. | E | TS | All |
| 19 | Marketing copy/ad audit vs prior period; **recommend, do not execute** budget shifts. | M | D | All |
| 20 | Repurpose one approved asset into N channel variants (LinkedIn / email subject lines). | E | TS | CG/PPX |

### Tier 4 — Capability ceiling (app-building & engineering; nice-to-have for this buyer)
| Rank | Task | Diff | Tag | Src |
|---|---|---|---|---|
| 21 | Build + deploy an internal dashboard from live data at a private, auth-gated link. | H | D | All |
| 22 | Commission/AUM-fee tracker as a deployed web app with login. | H | D | Claude |
| 23 | Overnight alert triage (monitoring + issue tracker) → morning summary + suggested actions. | M | D | All |
| 24 | Clone repo, fix a bug, open a PR (with human merge approval). | H | D | CL/CG |

### Eval design notes
- **Always run the action variants against the gate**, not just the happy path — for a fund, *what it refuses to do unsupervised* matters more than what it produces.
- **Seed at least one known-wrong number** in Tier-1 #4; the differentiator is whether the tool catches it or confidently passes it through.
- **Run Tier-1 #5 (cost) last**, after you know your real task cadence from Tiers 2–3, so the credit estimate reflects actual usage.
- Tasks 9, 11, and 1 are the same archetype (generate → gate → human sign-off). If a competitor nails that loop with an audit trail, it neutralizes most of Viktor's lead for a regulated buyer regardless of raw capability parity.

---

## 7. One-paragraph verdict

The corroborated core is solid and not marketing fluff: Viktor is a Slack-native agent that turns one message into a finished, tool-sourced deliverable behind an approval gate, priced on credits, strongest in marketing/finance/ops reporting, with **no proven asset-manager deployment**. The genuine uncertainty — and the entire competitive opening — sits in exactly the cells where the three reports *disagree*: enterprise governance (RBAC, retention, residency, audit depth), Teams parity, and true cost at scale. Perplexity is the source to discount where it reports those as shipped; ChatGPT is the source to remember is 100% vendor-self-description; Claude is the richest but carries the most uncorroborated single-source detail (including a possibly-confabulated "Zeta Labs / formerly Jace AI" corporate identity). Verify C1–C8 in writing before any of it goes in a deck.
