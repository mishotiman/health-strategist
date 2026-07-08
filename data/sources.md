# Seed corpus — sources

13 open-access papers spanning training, recovery, sleep, supplements, and
stress. **Every source is CC BY** (verified via the PMC OA service / publisher).
The source files themselves live in `data/papers/` and are **gitignored** to keep
the repo light — this manifest is the tracked record. Exact citation metadata
(authors, journal, year, DOI) is parsed from each JATS XML during ingestion.

Files are full-text **JATS XML** from the Europe PMC REST API, except #3 which is
a **PDF** preprint (SportRxiv). Both formats are handled by the ingestion step.

| # | Topic | Title | Source | License |
|---|-------|-------|--------|---------|
| 1 | Protein → strength | Synergistic Effect of Increased Total Protein Intake and Strength Training on Muscle Strength: A Dose-Response Meta-analysis of RCTs | [PMC9441410](https://pmc.ncbi.nlm.nih.gov/articles/PMC9441410/) · Sports Med Open, 2022 | CC BY |
| 2 | Sleep → performance | The Impact of Sleep Interventions on Athletic Performance: A Systematic Review | [PMC10354314](https://pmc.ncbi.nlm.nih.gov/articles/PMC10354314/) · Sports Med Open, 2023 | CC BY |
| 3 | Training volume/frequency → hypertrophy & strength | The Resistance Training Dose-Response: Meta-Regressions Exploring the Effects of Weekly Volume and Frequency (Pelland et al.) | [SportRxiv #460](https://sportrxiv.org/index.php/server/preprint/view/460) · 2024 | CC BY 4.0 |
| 4 | Aerobic base → VO₂max | The Effect of Polarized Training Intensity Distribution on Maximal Oxygen Uptake and Work Economy Among Endurance Athletes | [PMC11679080](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679080/) · Sports (MDPI), 2024 | CC BY |
| 5 | HRV → recovery | Heart Rate Variability-Guided Training for Enhancing Cardiac-Vagal Modulation, Aerobic Fitness, and Endurance Performance | [PMC8507742](https://pmc.ncbi.nlm.nih.gov/articles/PMC8507742/) · 2021 | CC BY |
| 6 | Supplement: creatine | Creatine for Exercise and Sports Performance, with Recovery Considerations for Healthy Populations | [PMC8228369](https://pmc.ncbi.nlm.nih.gov/articles/PMC8228369/) · Nutrients, 2021 | CC BY |
| 7 | Supplement: caffeine | International Society of Sports Nutrition Position Stand: Caffeine and Exercise Performance | [PMC7777221](https://pmc.ncbi.nlm.nih.gov/articles/PMC7777221/) · JISSN, 2021 | CC BY |
| 8 | Supplement: omega-3 | Omega-3 Supplementation on Post-Exercise Inflammation, Muscle Damage, Oxidative Response, and Sports Performance: A Systematic Review of RCTs | [PMC11243702](https://pmc.ncbi.nlm.nih.gov/articles/PMC11243702/) · Nutrients, 2024 | CC BY |
| 9 | Cold water immersion (dosing) | Impact of Different Doses of Cold Water Immersion on Recovery from Acute Exercise-Induced Muscle Damage: A Network Meta-Analysis | [PMC11897523](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897523/) · Front Physiol, 2025 | CC BY |
| 9b | Cold water immersion (hypertrophy caveat) | Throwing Cold Water on Muscle Growth: A Systematic Review with Meta-Analysis of Postexercise CWI on RT-Induced Hypertrophy | [PMC11235606](https://pmc.ncbi.nlm.nih.gov/articles/PMC11235606/) · 2024 | CC BY |
| 10 | Vitamin D | Effects of Vitamin D3 Supplementation on Strength of Lower and Upper Extremities in Athletes: An Updated Systematic Review and Meta-Analysis | [PMC11163122](https://pmc.ncbi.nlm.nih.gov/articles/PMC11163122/) · 2024 | CC BY |
| 11 | Magnesium (sleep) | Oral Magnesium Supplementation for Insomnia in Older Adults: A Systematic Review & Meta-Analysis | [PMC8053283](https://pmc.ncbi.nlm.nih.gov/articles/PMC8053283/) · 2021 | CC BY |
| 12 | Stress ↔ performance/health | Stress and Sport Performance: A PNEI Multidisciplinary Approach | [PMC10940545](https://pmc.ncbi.nlm.nih.gov/articles/PMC10940545/) · 2024 | CC BY |

## Notes on evidence nuance (for grounded, honest answers)

- **#9 / #9b (cold water):** good for acute recovery / soreness, but immersion
  *right after resistance training* can blunt hypertrophy — the agent should
  surface this trade-off rather than recommend ice baths unconditionally.
- **#8 (omega-3):** solid for recovery / muscle-damage markers, weak for direct
  performance gains.
- **#10 (vitamin D):** effect mainly on lower-body strength; mixed for upper body
  and power.
