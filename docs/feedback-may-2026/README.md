# Platform Feedback — Shiri, May 2026

Source: email from Shiri Horn-Brezel to Alexey Geht on 2026-05-13. Shiri reported
multiple issues across the dashboard after a working session. The original email
is in Hebrew; this folder is the engineering-side trace.

Each topic below is its own document so each can land as a separate PR that
Codex reviews independently. Where the gap is unambiguous (porting an existing
widget, adding a missing filter to a tab that should have had it) the
accompanying code change ships in the same PR. Where the right answer is a
judgement call we want Codex / Alexey to weigh in on, the document is
**research-only** with numbered options.

| # | Topic | Doc | Code? |
|---|---|---|---|
| A | Enrich tab — partial-batch behaviour (Crustdata + emails) | [`enrich-ux.md`](enrich-ux.md) | no |
| B | Filter ↔ Filter+ overlap + tenure-filter parity + OR/AND scope | [`filter-overlap.md`](filter-overlap.md) | yes (port tenure widget) |
| C | AI screening — skills, tenure verdict drift, "Lead" downgrade | [`screening-audit.md`](screening-audit.md) | no |
| D | Career-break / maternity-leave detection | [`career-break-detection.md`](career-break-detection.md) | no |
| E | DB-search — sparse-profile filter + tenure-filter parity | [`db-search-quality.md`](db-search-quality.md) | yes (filters) |

## How to read each doc

Every document follows the same shape:

1. **What Shiri reported** — one paragraph, paraphrased into English.
2. **What the code does today** — quoted snippets + line refs.
3. **Findings / hypotheses** — what we think is going on, with evidence.
4. **Options** — numbered, each with trade-offs. The PR description picks one
   to ship (if any), the others stay on file for Codex to challenge.
5. **Open questions** — anything we need to ask Shiri before deciding.

## Branches / PRs

Per `CLAUDE.md`, each topic ships on a feature branch named `<type>/<slug>`
against `master`. The branch names match the table above (A–E). All work is
prepared on `claude/review-platform-feedback-SEvMN` first so each topic can be
cherry-picked or branched off cleanly.
