"""04_glm.py
Fits the two standard pricing GLMs on the training sample:
  frequency: Poisson, log link, log(exposure) offset   -> claims per policy year
  severity:  Gamma, log link, weighted by claim count   -> average capped claim size
Pure premium = predicted frequency x predicted severity x large loss loading.
Saves the coefficient tables and a prediction for every policy (train and test) for later steps.

Run from the project folder:  python scripts/04_glm.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from common import CLEAN, LARGE_LOSS_LOADING, TABLES, load_policies

# Area is left out: it is built from population density, which is already in the model (see decisions.md)
FREQ_VARIABLES = ["drv_age_band", "bm_band", "veh_age_band", "veh_power_band",
                  "density_band", "VehGas", "VehBrand", "Region"]
# Severity keeps only driver age: in tests the other variables added nothing significant (see decisions.md)
SEV_VARIABLES = ["drv_age_band"]


def base_levels(df):
    """Base (reference) level for each variable = the level with the most exposure, a common pricing convention."""
    return {v: df.groupby(v)["exposure"].sum().idxmax() for v in FREQ_VARIABLES}


def formula(target, variables, bases):
    terms = [f"C({v}, Treatment('{bases[v]}'))" for v in variables]
    return f"{target} ~ " + " + ".join(terms)


def fit_severity(claimants, variables, bases):
    return smf.glm(formula("avg_severity", variables, bases), data=claimants,
                   family=sm.families.Gamma(link=sm.families.links.Log()),
                   var_weights=claimants["claim_count"]).fit()


def variable_tests(freq, claimants, bases):
    """Does each variable earn its place? Wald tests on the frequency GLM; likelihood ratio
    tests on a severity GLM with all variables (drop one variable at a time)."""
    rows = []
    wald = freq.wald_test_terms(skip_single=False, scalar=True).summary_frame()
    for v in FREQ_VARIABLES:
        r = wald.loc[[i for i in wald.index if i.startswith(f"C({v},")][0]]
        rows.append(("frequency", v, int(r["df constraint"]), r["chi2"], r["P>chi2"]))
    full = fit_severity(claimants, FREQ_VARIABLES, bases)
    for v in FREQ_VARIABLES:
        reduced = fit_severity(claimants, [x for x in FREQ_VARIABLES if x != v], bases)
        stat = (reduced.deviance - full.deviance) / full.scale
        df_diff = int(full.df_model - reduced.df_model)
        rows.append(("severity", v, df_diff, stat, stats.chi2.sf(stat, df_diff)))
    return pd.DataFrame(rows, columns=["model", "variable", "df", "chi_square", "p_value"])


def coefficient_table(result, variables, bases, model):
    ci = result.conf_int()
    rows = []
    for name, coef in result.params.items():
        if name == "Intercept":
            variable, level = "Intercept", "base"
        else:
            variable = name.split(",")[0].replace("C(", "")
            level = name.split("[T.")[1].rstrip("]")
        rows.append({
            "model": model, "variable": variable, "level": level,
            "coefficient": coef, "relativity": np.exp(coef),
            "ci_low": np.exp(ci.loc[name, 0]), "ci_high": np.exp(ci.loc[name, 1]),
            "p_value": result.pvalues[name],
        })
    for v in variables:   # base levels have relativity 1 by definition
        rows.append({"model": model, "variable": v, "level": bases[v], "coefficient": 0.0,
                     "relativity": 1.0, "ci_low": 1.0, "ci_high": 1.0, "p_value": np.nan})
    return pd.DataFrame(rows)


def main():
    df = load_policies()
    train = df[df["sample"] == "train"].copy()
    bases = base_levels(train)
    print("Base levels (most exposure):", bases, "\n")

    # Frequency: Poisson with log(exposure) offset
    freq = smf.glm(formula("claim_count", FREQ_VARIABLES, bases), data=train,
                   family=sm.families.Poisson(), offset=np.log(train["exposure"])).fit()

    # Severity: Gamma on each policy's average capped claim, weighted by its number of claims
    claimants = train[train["claim_count"] > 0].copy()
    claimants["avg_severity"] = claimants["claim_amount_capped"] / claimants["claim_count"]
    sev = fit_severity(claimants, SEV_VARIABLES, bases)

    tests = variable_tests(freq, claimants, bases)

    coefs = pd.concat([coefficient_table(freq, FREQ_VARIABLES, bases, "frequency"),
                       coefficient_table(sev, SEV_VARIABLES, bases, "severity")], ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tests.round(4).to_csv(TABLES / "glm_variable_tests.csv", index=False)
    coefs.round(6).to_csv(TABLES / "glm_coefficients.csv", index=False)

    # Predictions for every policy, train and test
    df["pred_frequency"] = freq.predict(df, offset=np.zeros(len(df)))   # claims per policy year
    df["pred_severity"] = sev.predict(df)
    df["pred_pure_premium"] = df["pred_frequency"] * df["pred_severity"] * LARGE_LOSS_LOADING
    df[["policy_id", "sample", "pred_frequency", "pred_severity", "pred_pure_premium"]] \
        .to_parquet(CLEAN / "glm_predictions.parquet", index=False)

    # Fit summary and checks
    print(f"Frequency GLM: {len(train):,} policies, {int(freq.df_model)} parameters, "
          f"deviance {freq.deviance:,.0f}, AIC {freq.aic:,.0f}")
    print(f"Severity GLM:  {len(claimants):,} policies with claims, {int(sev.df_model)} parameters, "
          f"dispersion {sev.scale:.3f}\n")
    checks = []
    for s in ["train", "test"]:
        d = df[df["sample"] == s]
        actual_claims = d["claim_count"].sum()
        pred_claims = (d["pred_frequency"] * d["exposure"]).sum()
        actual_loss = d["claim_amount"].sum()
        pred_loss = (d["pred_pure_premium"] * d["exposure"]).sum()
        checks += [
            (s, "Actual claims", actual_claims), (s, "Predicted claims", round(pred_claims, 1)),
            (s, "Predicted / actual claims", round(pred_claims / actual_claims, 4)),
            (s, "Actual losses (uncapped)", round(actual_loss, 0)), (s, "Predicted losses", round(pred_loss, 0)),
            (s, "Predicted / actual losses", round(pred_loss / actual_loss, 4)),
        ]
    checks = pd.DataFrame(checks, columns=["sample", "check", "value"], dtype=object)
    print(checks.to_string(index=False))
    checks.to_csv(TABLES / "glm_checks.csv", index=False)

    print("\n--- Does each variable earn its place? (p below 0.05 = yes) ---")
    print(tests.round(4).to_string(index=False))

    show = coefs[coefs["variable"] != "Intercept"].copy()
    print("\n--- Frequency relativities (exp of coefficient), 95% interval ---")
    f = show[show["model"] == "frequency"].sort_values(["variable", "level"])
    print(f[["variable", "level", "relativity", "ci_low", "ci_high", "p_value"]].round(3).to_string(index=False))
    print("\n--- Severity relativities ---")
    s = show[show["model"] == "severity"].sort_values(["variable", "level"])
    print(s[["variable", "level", "relativity", "ci_low", "ci_high", "p_value"]].round(3).to_string(index=False))
    print(f"\nSaved {TABLES / 'glm_coefficients.csv'}, {TABLES / 'glm_checks.csv'}, {TABLES / 'glm_variable_tests.csv'} and {CLEAN / 'glm_predictions.parquet'}")


if __name__ == "__main__":
    main()
