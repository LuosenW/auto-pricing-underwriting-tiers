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
