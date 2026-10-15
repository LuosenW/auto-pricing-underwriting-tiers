# Underwriting guidelines: motor third party liability

Draft guidelines based on the GLM rating plan in this project. Synthetic portfolio work for a portfolio project, not advice for a real book.

## Summary

Under one flat rate, the safest 42% of policies would run a 37% loss ratio while the riskiest 1.4% would run 359%. The GLM plan brings every tier close to the 70% target. Five tiers, set by each policy's predicted cost relative to the book average, decide how a policy is handled.

## Tiers

| Tier | Predicted cost vs average | Share of policies | Share of losses | Typical profile | Action |
| --- | --- | --- | --- | --- | --- |
| 1 Preferred | under 0.75 | 42% | 24% | Average age 50; 94% at the best bonus malus (50) | Accept. Best rates; focus retention here, since these customers are the most profitable and the most likely to be won away by a competitor. |
| 2 Standard | 0.75 to 1.25 | 32% | 31% | Average age 46; about half at bonus malus 50 | Accept at plan rate. |
| 3 Surcharge | 1.25 to 2 | 18% | 23% | Average age 38; average bonus malus 74 | Accept at the GLM rate, which carries the surcharge. No review needed. |
| 4 Refer | 2 to 4 | 6% | 17% | Average age 38; 41% at bonus malus 100 or above; 15% in the densest areas | Refer to an underwriter before binding. Check the claims history (loss runs) and the driving record, confirm who drives the car, and consider a higher deductible. |
| 5 Decline | 4 and over | 1.4% | 5% | Average age 30; 65% aged 25 or under; 90% at bonus malus 100 or above; 4.5 times the average claim frequency | Do not write at standard terms. Senior underwriter sign off required. Third party liability is compulsory in France, so in practice this tier is written only with conditions (higher deductible, restricted named drivers, telematics) or at the full GLM rate. |

Loss ratios are measured on the test sample (118,358 policies the models never saw), with claims capped at 100,000 times the large loss loading, at an assumed 70% target loss ratio.

| Tier | Loss ratio, one flat rate | Loss ratio, GLM plan |
| --- | --- | --- |
| 1 Preferred | 37% | 62% |
| 2 Standard | 70% | 75% |
| 3 Surcharge | 113% | 72% |
| 4 Refer | 246% | 94% |
| 5 Decline | 359% | 63% |

## Notes and caveats

- The Refer tier runs a 94% loss ratio even under the GLM plan. With 582 claims in the test sample this is within normal claim size noise, but it supports the referral rule and is the first tier to watch as more data comes in.
- The Preferred tier runs 62% under the GLM plan, a little below target, so there is some room to sharpen rates for the best customers if competition requires it.
- Tier cutoffs (0.75, 1.25, 2 and 4 times average) are judgment calls chosen to give a large Preferred group and a Decline group near 1% of policies. They should be revisited with the business.
- The model only sees the 8 rating variables. An underwriter reviewing a Refer or Decline case can use information the model does not have, such as prior insurer, driving convictions and how the car is used.
