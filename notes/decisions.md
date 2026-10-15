# Data decisions log

## Source
- freMTPL2freq (OpenML data_id 41214) and freMTPL2sev (OpenML 41215): French motor third party liability policies and claims, loaded with scikit-learn's fetch_openml.

## Raw load checks (01_get_data.py)
- 678,013 policies, all with unique IDs; 358,499 policy years of exposure.
- 26,639 claim amount rows on 24,950 policies; total claim amounts 60,697,931 (euros). The freq file reports 36,102 claims.
- No missing values in either file; no claim amounts of 0 or less.
- Text column VehGas is stored with quote marks ('Regular', 'Diesel'). Fix: strip the quotes.
- 1,224 policies show exposure above 1 year (maximum 2.01). Policies are annual, so exposure is capped at 1.
- 195 claim amount rows (788,714) belong to policy IDs that are not in the freq file. They cannot be rated, so they are dropped and reported.
- Large claims: 42 claims above 100,000 make up 24.6% of all claim dollars; the largest is 4,075,401. To be capped before the severity model.
- About a third of claim amounts sit at a few fixed values (1,204 on 4,798 claims, 1,128 on 3,896 claims), which looks like flat settlement amounts. Noted for the severity model.

## Claim counts without claim amounts
- 9,116 policies (27% of the 34,060 policies with a claim) report claims but have no claim amount.
- They are almost all in one block of policy IDs: IDs up to 1,000,000 hold 87,030 policies (15% of exposure). In that block 68% of claiming policies have no amount; elsewhere it is under 1%.
- That block's reported claim frequency is 0.269 claims per policy year, 3.7 times the 0.072 of all other policies, even though its mix of driver age, bonus malus, vehicle age and area is about the same. Counting only claims with amounts, the block's frequency is 0.086.
- Conclusion so far: the claim counts in that block look unreliable, not just the amounts.
- Options considered: (1) drop the ID block; (2) keep all policies but count only claims with amounts (the scikit-learn tutorial approach); (3) keep reported counts for frequency and use claims with amounts for severity.
- Decision (Luosen, Oct 15): option 1, drop policy IDs up to 1,000,000. The block's counts look wrong, not just its amounts, so keeping it would push frequency up for the whole book. Dropping it keeps claim counts and claim amounts consistent, so modeled losses tie back to actual losses for the loss ratio and actual versus expected work. Cost: 87,030 policies (13%) and 52,897 policy years (15% of exposure).

## Cleaning results (02_clean.py)
- Dropped and reported: 195 claim rows (788,714) with no matching policy; 87,030 policies, 4,566 claim rows and 10,239,627 of claim amounts in the dropped ID block.
- Exposure capped at 1 year on 1,224 policies.
- Outside the block, 18 policies still had a claim count that did not match their claim amounts (17 with no amount at all). Their claim count was set to the number of claims with amounts.
- Final data: 590,983 policies, 305,463 policy years, 21,878 claims, 49,669,589 of claim amounts. Raw claim amounts minus kept minus dropped = 0.00.
- Large claims: each claim is capped at 100,000 for the severity model (34 claims, 0.16% of claims). Capped losses are 82.1% of total losses, so a flat large loss loading of 1.2182 (total / capped) is applied to modeled premiums. Thresholds compared: 50,000 keeps 77.5% of losses, 100,000 keeps 82.1%, 200,000 keeps 87.1%; 100,000 was chosen as a round figure between the 99.9th percentile (181,308) and the point where the severity model would be driven by a handful of claims.
- Train and test split: 80% / 20% by policy, random seed 42. Train: 472,625 policies, claim frequency 0.0715. Test: 118,358 policies, frequency 0.0722.

## One-way analysis (03_one_way.py)
- Run on the training sample only (472,625 policies), so the test sample stays untouched for validating the models.
- Bands are defined once in scripts/common.py and reused by every later step: driver age (8 bands), bonus malus (6), vehicle age (6), vehicle power (7), density (5). Area, fuel, brand and region are used as coded.
- Book average (capped): frequency 0.0715 claims per policy year, severity 1,844, pure premium 131.80 per policy year.
- Thin cells to watch: bonus malus 126+ (0.1% of exposure, relativity 10.3), driver age 18 to 20 (0.7%, relativity 5.3), several regions under 1% of exposure. One-way relativities on cells this small are noisy; the GLM and later grouping decide how much of them to trust.
- One-way results overlap: area and density measure nearly the same thing, and young drivers also tend to have high bonus malus levels. A one-way view counts that overlap twice; the GLM separates it.

## GLMs (04_glm.py)
- Frequency: Poisson GLM, log link, log(exposure) offset, on the training sample (472,625 policies). Severity: Gamma GLM, log link, on each claiming policy's average capped claim, weighted by its number of claims (16,428 policies, 17,474 claims).
- Base level for each variable = the level with the most exposure (driver age 41-50, bonus malus 50, vehicle age 6-10, power 6, density 201-1,000, Regular fuel, brand B1, region R24).
- Area is left out of both models. It is a grouping of population density, which is already in the model; keeping both would split one effect across two correlated variables.
- Variable tests (output/tables/glm_variable_tests.csv). Frequency: all 8 variables significant (p < 0.001). Severity, with all 8 variables in: driver age p = 0.034, bonus malus p = 0.17, all others p between 0.56 and 0.99.
- Decision (Luosen, Oct 15): severity model uses driver age only. Options were driver age only, all 8 variables, or flat severity. Driver age is the only variable with evidence behind it; keeping the rest would put noise into the rates (for example region severity relativities driven by a few large claims). Severity for drivers 18 to 20 is 1.84 times the base (95% interval 1.38 to 2.47).
- Checks: predicted claims equal actual claims on the training sample (a property of a Poisson GLM with an intercept); on the test sample predicted / actual claims = 0.990. Predicted / actual losses: 0.976 train, 1.037 test.
- Confounding shown by the GLM: one-way, drivers 18 to 20 have 5.3 times average pure premium; in the frequency GLM their relativity is only 1.12 once bonus malus is in the model, and drivers 21 to 30 come out below the 41 to 50 base. New French drivers start at bonus malus 100, so bonus malus already carries most of the young driver effect. Worth a note in the rating plan step: a pricing team might smooth or constrain the driver age curve so it does not dip for drivers in their 20s.

## Rating plan (05_rating_plan.py)
- Plan form: pure premium = base rate x one relativity per variable (driver age, bonus malus, vehicle age, vehicle power, density, fuel, brand, region). Each relativity = frequency relativity x severity relativity (severity only varies by driver age), rounded to 3 decimals as a filed rate table would be.
- Base rate: 104.64 from the models (base frequency x base severity x large loss loading 1.2182), times an off-balance factor of 1.0248 so the plan reproduces actual uncapped training losses. Final base rate 107.23 per policy year.
- Checks: plan premium / actual losses = 1.000 on train and 1.063 on test (the test sample happened to have lighter losses; actual versus expected by segment is checked in the validation step). Plan premium ranges from 32 to 3,081 per policy year; the middle 90% of policies fall between 77 and 387.
- Driver age curve kept as fitted, not smoothed. Combined driver age relativities are 2.07 (18 to 20), 0.92 (21 to 25), 0.71 (26 to 30), 0.82 (31 to 40), 1.00 (41 to 50). The dip under 41 to 50 is not noise: French drivers start at bonus malus 100 and need years without claims to reach 50, so a driver in their 20s almost always also carries a bonus malus surcharge (2.4 times at 61 to 80). The age relativity offsets part of that surcharge. Smoothing the age curve upward would charge young drivers twice for the same risk. A real filing would still show regulators the combined age and bonus malus effect for typical young drivers.

## Validation (06_validate.py), test sample of 118,358 policies
- "Actual" losses for validation = claims capped at 100,000 x large loss loading 1.2182, so a few huge claims do not decide the comparison. Every alternative is rebalanced to training losses the same way the GLM plan is.
- Alternatives compared: one-way plan (base rate x one-way relativities), a Tweedie GLM fit directly on pure premium (power 1.5, same variables), and gradient boosted trees (scikit-learn, Poisson loss, raw variables). The Tweedie benchmark uses scikit-learn's TweedieRegressor because statsmodels ran out of memory on this one in a 4 GB test environment.
- Gini on the test sample: one-way plan 0.297, GLM rating plan 0.340, Tweedie GLM 0.348, boosted trees 0.335. The GLM plan ranks risk about 14% better than the one-way plan; neither benchmark beats it by a meaningful margin, so the simpler, explainable GLM plan is kept.
- Double lift: grouped by GLM premium / one-way premium, the one-way plan's actual / premium runs from 0.49 to 2.28; the GLM plan's stays between 0.88 and 1.20.
- Lift chart: the top tenth of policies by GLM premium cost 473 per policy year actual versus 430 predicted; the bottom tenth 51 versus 74. Middle deciles are noisier.
- Actual versus expected by segment: first checked on losses, which flagged 24 segments outside 0.85 to 1.15, mostly claim size noise (a few hundred claims with a severity dispersion near 8). Switched the main test to claim counts with a 95% band for Poisson noise: 5 of 67 segments fall outside the band, close to the 3 to 4 expected by chance (vehicle age 6-10 at 1.06, vehicle power 8 at 0.84, density 201-1,000 at 1.10, brand B6 at 1.16, region R82 at 1.10). No variable shows a pattern across neighboring levels, so no refit.

## Impact analysis (07_impact.py), all 590,983 policies
- Assumption: the insurer's current plan is the one-way plan (base rate x one-way relativities). The real current rates are not in the data. Both plans are set to collect the same total premium (50,260,385 on this year's exposure), so the study shows only who pays more and who pays less. Pure premium basis; an expense and profit load applied to both would not change any percentage.
- Full move to the GLM plan: 71.5% of policies go up, 28.5% go down; 75.5% change by more than 15% either way; median change +21.7%; range -78% to +306%.
- Why so many increases: the one-way plan multiplies 8 relativities that overlap (young drivers also have high bonus malus, dense areas also have certain regions and brands), so it badly overcharges a minority of high risk policies and, to collect the same total, undercharges the safe majority. The GLM undoes this: drivers 18 to 20 average -56%, bonus malus 126+ -52%, bonus malus 100-125 -41%; drivers 71+ +63%, 61-70 +47%, bonus malus 50 (57% of policies) +34%. The one-way plan's double lift result (actual / premium from 0.49 to 2.28) is the same problem seen from the loss side.
- Capping: changes limited to plus or minus 15% this renewal, with the GLM rates inside the cap scaled by 1.023 so the capped plan still collects the same total. 77% of policies sit at the cap; 69% get an increase of 5% to 15% and 23% a decrease of 5% to 15%. With a 15% cap each year, the median policy (+22%) reaches the GLM rate at the second renewal.
- Fix along the way: policies capped at exactly -15% were first counted in the "Down 15-25%" band (pd.cut includes the right edge). Changes are nudged a hair toward zero before banding so capped policies land in the 5-15% bands.

## Underwriting tiers (08_tiers.py)
- Tiers are set by GLM predicted cost relative to the book average pure premium (164.54 per policy year): Preferred under 0.75, Standard 0.75 to 1.25, Surcharge 1.25 to 2, Refer 2 to 4, Decline 4 and over. Cutoffs are judgment calls: they give a large Preferred tier (42% of policies) and a Decline tier near 1% (1.4%).
- Premium assumes a 70% target loss ratio (premium = pure premium / 0.70). Loss ratios are measured on the test sample with capped claims x large loss loading.
- Result: under one flat rate, tier loss ratios run from 37% (Preferred) to 359% (Decline); under the GLM plan they run from 62% to 94%, with the overall test loss ratio at 72.5% under both.
- The Refer tier's 94% under the GLM plan rests on 582 claims; with a severity dispersion near 8 that is within noise, so no change to the plan, but it is flagged in the underwriting guidelines.
- Guidelines memo: notes/underwriting_guidelines.md.

## Tableau export (09_export_tableau.py)
- Results are reshaped into 5 CSVs in dashboard/tableau_data/ (committed, small): segments.csv, tiers.csv, impact_distribution.csv, model_comparison.csv, lift_chart.csv.
- segments.csv holds one row for each of the 67 rating variable levels, with the GLM and one-way relativities, claim count actual versus expected (test sample) and average premium change. Keeping them in one file lets a single Variable filter drive every segment chart without Tableau data blending.
- Each file has an order column (level_order, tier_order, band_order) so Tableau shows bands in their natural order. Brands and regions are ordered by GLM relativity.
