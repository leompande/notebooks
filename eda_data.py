import pandas as pd
import numpy as np

def make_churn(n=500, seed=42):
    rng = np.random.default_rng(seed)
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    region = rng.choice(["Dar es Salaam", "Arusha", "Mwanza", "Dodoma", "Mbeya"], n, p=[.4, .2, .2, .1, .1])
    plan = rng.choice(["basic", "standard", "premium"], n, p=[.5, .35, .15])
    tenure = rng.gamma(2.0, 8.0, n).round(1)                      # right-skewed months
    monthly = rng.normal(25, 6, n) + (plan == "premium") * 20 + (plan == "standard") * 8
    usage = np.exp(rng.normal(3, 0.8, n))                          # log-normal
    tickets = rng.poisson(1 + (tenure < 6) * 2, n)
    logit = -1.5 - 0.06 * tenure + 0.35 * tickets + 0.02 * (monthly - 30)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(dict(signup=signup, region=region, plan=plan, tenure=tenure,
                           monthly_fee=monthly.round(2), usage_gb=usage.round(2),
                           support_tickets=tickets, churned=churn.astype(int)))
    df.loc[rng.random(n) < 0.08, "usage_gb"] = np.nan  # missing values
    df.loc[rng.choice(n, 15, replace=False), "monthly_fee"] *= 10  # injected outliers
    return df.sort_values("signup").reset_index(drop=True)
