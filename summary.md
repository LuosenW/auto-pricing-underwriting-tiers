# Pricing car insurance fairly: a one page summary

**Luosen Wei** | Auto insurance pricing and underwriting project | French motor liability data, 590,983 policies

## The headline

A pricing model that looks at all of a driver's risk factors together charges customers much closer to what they actually cost than simple one factor at a time rates, and shows that a single flat price would make the safest drivers pay for the riskiest.

## The question

An insurer needs to decide what each customer should pay. Charge too much and good customers leave; charge too little and the insurer loses money on them. I built a pricing model from about 590,000 real (anonymized) French car insurance policies and tested it on policies it had never seen.

## What I found

- **Looking at one factor at a time double counts risk.** Young drivers look almost 6 times as costly as drivers in their 40s. But young drivers also have no claim free history yet, which insurers already charge for. Counting both at once, the true extra cost of youth alone is about 2 times.
- **The model sorts good and bad risks about 14% better** than the simpler approach. I also tried a machine learning model; it was no better, so I kept the model that is easy to explain to a regulator or a manager.
- **How often people crash is predictable; how much a crash costs mostly is not.** Claim history, age and where people live all predict how often someone claims. Only driver age had any real link to the size of a claim.
- **Switching to the new prices would move most customers.** About 7 in 10 would pay more and 3 in 10 less, with the same total premium. Young and high risk drivers would pay about half as much as before; drivers over 70 would pay about 60% more. Limiting changes to 15% a year spreads this over two renewals.
- **One flat price would hide a large cross subsidy.** At a single price, the safest 42% of customers would cost the insurer 37 cents in claims for every euro of premium, and the riskiest 1.4% would cost 3.59 euros. Under the model's prices every group lands between 62 and 94 cents per euro, close to the 70 cent target.

![Loss ratio by underwriting tier](output/charts/07_underwriting_tiers.png)

## What an insurer could do with this

- **Use five underwriting tiers.** Accept the Preferred, Standard and Surcharge tiers at the model price, send the Refer tier (6% of policies) to an underwriter before binding, and only write the Decline tier (1.4%) with conditions such as a higher deductible.
- **Phase in the new prices.** Cap each customer's change at 15% a year to keep customers who would otherwise face a big jump.
- **Watch the Refer tier.** It still runs above target under the new prices, so it is the first group to recheck as new claims come in.

## About the data

The data is public, anonymized French car insurance data widely used for teaching pricing. I removed one block of 87,030 policies whose claim records were incomplete, and capped very large claims so a handful of them could not distort the results. Every choice is written down with its reasoning in this repository.
