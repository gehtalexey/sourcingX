**The missing piece is what evidence you accept for outreach.** Kalamata’s Dwelly results reflect preferences learned from you, including exceptions for company background, degrees and short jobs. A generic rule cannot reproduce preferences absent from the typed brief.

**I recommend your four states, with a stricter definition of “implied” and no separate score threshold.** This is a design recommendation; its accuracy still needs testing.

The rule in three lines:

1. Reject a demonstrated contradiction, matched exclusion or applicable hard-filter failure.
2. Otherwise, approve only when every must-have has direct evidence or a specific, defensible inference.
3. Reject remaining unknowns as “insufficient evidence for outreach”; record inferred items as questions for the call.

Use these exact states:

| State | Meaning |
|---|---|
| `met` | Evidence directly supports the requirement at the level requested. |
| `implied` | Specific career evidence supports that requirement without stating it directly. |
| `not_met` | Specific evidence contradicts the requirement. |
| `unproven` | Neither direct evidence nor a defensible inference supports it. |

The boundary matters more than the number of states. “Senior Full Stack Engineer” supports experience across frontend and backend; it does **not** establish React, Python or five years using either. An engineering role at an AI company supports production AI only when the available product/team evidence connects that person’s work to AI, or the brief explicitly accepts that company route.

If you want Dwelly’s broader exception, the brief must say: **“An engineering role at an AI-native product company counts; written AI project details are not required.”** Otherwise SourcingX would be inventing your hiring preference.

Use this prompt wording:

> Evaluate whether the evidence justifies outreach, not whether every qualification is proven for hiring.
>
> For each must-have, first find direct evidence. If absent, assess indirect evidence tied specifically to that requirement. For `implied`, cite the observed fact and explain why it supports the requirement.
>
> Relevant titles can support broad responsibilities. Employer prestige, generic seniority and unrelated strengths cannot establish a specific technology, specialist experience, qualification or numeric threshold. Never invent company or team facts.
>
> Interpret alternatives literally. Apply any substitute or evidence route explicitly accepted in the brief. Do not silently relax a requirement.
>
> Missing detail is not a contradiction. Use `unproven` when support remains insufficient; explain “insufficient evidence of X,” never “candidate lacks X.”
>
> Do not lower an otherwise passing decision merely because its accepted evidence is indirect.

**Let the code derive GO/NO GO from those verdicts.** Require a valid answer for every criterion; missing answers must never silently pass. Use the score only to rank candidates: SourcingX currently describes it as independent of the decision, and the general Kalamata rulebook permits 6 while Dwelly requires 7. Neither establishes a universal threshold.

Store GO as `Good Fit`, NO GO as `Not a Fit`, and inferred items in `screening_notes`. This needs no additional model call.

Before testing, reconcile the existing “full profile required” instruction with accepting thin profiles. Also resolve SourcingX’s compulsory stability rejection versus Dwelly’s explicit exception; changing evidence states cannot fix that disagreement.

**The one settling check:** freeze this rule and run the blind four-role comparison, reporting missed outreach candidates and wrongly approved rejects separately for each role. Keep the 20 Kalamata-approved people as a regression set, not proof of accuracy; the new rule succeeds only if it reduces missed outreach without exceeding a false-approval limit you choose before seeing results.