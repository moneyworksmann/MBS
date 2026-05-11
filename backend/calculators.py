"""
FutureFund — Financial Calculators Backend
===========================================
Modules:
  1. Monte Carlo (S&P 500) — year-by-year percentile paths + CAGR scenarios
  2. Monte Carlo (CPI) — inflation scenarios
  3. Debt repayment — Avalanche & Snowball with cascade
  4. Retirement planner — Traditional IRA vs Roth IRA (2024 limits)
  5. Investment growth — FV compound with monthly contributions
  6. Loan amortization — standard + extra payment
  7. Cash flow analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# HISTORICAL DATA  (1974–2024, 51-year window)
# ─────────────────────────────────────────────────────────────────

SP500_ANNUAL: Dict[int, float] = {
    1974:-25.90,1975:37.00,1976:23.60,1977:-7.40,1978:6.40,
    1979:18.20,1980:31.74,1981:-4.92,1982:21.41,1983:22.51,
    1984:6.27, 1985:32.16,1986:18.47,1987:5.23, 1988:16.81,
    1989:31.49,1990:-3.17,1991:30.55,1992:7.67, 1993:9.99,
    1994:1.33, 1995:37.43,1996:23.07,1997:33.36,1998:28.34,
    1999:20.89,2000:-9.10,2001:-11.89,2002:-22.10,2003:28.36,
    2004:10.74,2005:4.83, 2006:15.61,2007:5.48, 2008:-36.55,
    2009:25.94,2010:14.82,2011:2.10, 2012:15.89,2013:32.15,
    2014:13.52,2015:1.38, 2016:11.74,2017:21.64,2018:-4.23,
    2019:31.21,2020:18.02,2021:28.47,2022:-18.17,2023:26.06,
    2024:23.31,
}

CPI_ANNUAL: Dict[int, float] = {
    1974:11.06,1975:9.13,1976:5.74,1977:6.45,1978:7.63,
    1979:11.26,1980:13.55,1981:10.33,1982:6.13,1983:3.21,
    1984:4.30, 1985:3.56, 1986:1.86, 1987:3.65, 1988:4.14,
    1989:4.82, 1990:5.39, 1991:4.25, 1992:3.03, 1993:2.96,
    1994:2.61, 1995:2.81, 1996:2.93, 1997:2.34, 1998:1.55,
    1999:2.19, 2000:3.37, 2001:2.83, 2002:1.59, 2003:2.27,
    2004:2.68, 2005:3.39, 2006:3.24, 2007:2.85, 2008:3.85,
    2009:-0.36,2010:1.64, 2011:3.16, 2012:2.07, 2013:1.46,
    2014:1.62, 2015:0.12, 2016:1.26, 2017:2.13, 2018:2.44,
    2019:1.81, 2020:1.23, 2021:4.70, 2022:8.00, 2023:4.12,
    2024:2.90,
}

IRA_LIMIT_UNDER50          = 7_000
IRA_LIMIT_OVER50           = 8_000
ROTH_PHASEOUT_SINGLE_START = 146_000
ROTH_PHASEOUT_SINGLE_END   = 161_000
ROTH_PHASEOUT_MFJ_START    = 230_000
ROTH_PHASEOUT_MFJ_END      = 240_000
RMD_START_AGE              = 73


# ─────────────────────────────────────────────────────────────────
# 1 & 2.  MONTE CARLO
# ─────────────────────────────────────────────────────────────────

def run_monte_carlo_paths(
    initial: float,
    monthly: float,
    n_sim: int = 10_000,
    n_years: int = 30,
    annual_pct_data: Optional[Dict] = None,
) -> dict:
    """
    10,000-path log-normal Monte Carlo on S&P 500 historical returns.
    Returns year-by-year portfolio values at 5 percentiles for Chart.js fan chart.
    """
    if annual_pct_data is None:
        annual_pct_data = SP500_ANNUAL

    returns = np.array(list(annual_pct_data.values()))
    log_r   = np.log(1 + returns / 100)
    mu, sig = log_r.mean(), log_r.std()

    # Shape (n_sim, n_years) — one annual log-return per year per simulation
    sim      = np.random.normal(mu, sig, (n_sim, n_years))
    annual_r = np.exp(sim) - 1                        # arithmetic returns
    monthly_r = (1 + annual_r) ** (1.0 / 12) - 1     # geometric monthly equiv

    # Build portfolio value matrix (n_sim, n_years+1)
    portfolio      = np.zeros((n_sim, n_years + 1))
    portfolio[:, 0] = initial

    for y in range(n_years):
        mr     = monthly_r[:, y]                      # shape (n_sim,)
        pv     = portfolio[:, y]
        growth = (1 + mr) ** 12
        # FV = PV*(1+mr)^12 + C * ((1+mr)^12 - 1) / mr
        safe_mr = np.where(np.abs(mr) < 1e-9, 1e-9, mr)
        portfolio[:, y + 1] = pv * growth + monthly * (growth - 1) / safe_mr

    pcts  = [5, 25, 50, 75, 95]
    paths = {
        f"p{p}": [round(float(np.percentile(portfolio[:, y], p)), 0)
                  for y in range(n_years + 1)]
        for p in pcts
    }

    final = portfolio[:, -1]

    # Scalar CAGR scenarios for summary table
    cagr     = (np.exp(sim.mean(axis=1)) - 1) * 100
    scenarios = {
        "pessimistic":  round(float(np.percentile(cagr, 10)), 2),
        "conservative": round(float(np.percentile(cagr, 25)), 2),
        "median":       round(float(np.percentile(cagr, 50)), 2),
        "optimistic":   round(float(np.percentile(cagr, 75)), 2),
        "bull":         round(float(np.percentile(cagr, 90)), 2),
    }

    return {
        "years":            list(range(n_years + 1)),
        "paths":            paths,
        "final_stats":      {f"p{p}": round(float(np.percentile(final, p)), 0) for p in pcts},
        "scenarios":        scenarios,
        "total_contributed": round(initial + monthly * n_years * 12, 0),
        "n_sim":            n_sim,
        "n_years":          n_years,
        "hist_mean":        round(float(returns.mean()), 2),
        "hist_std":         round(float(returns.std()),  2),
    }


def run_monte_carlo_scenarios(annual_pct_data: dict, n_sim: int = 10_000,
                               n_years: int = 30) -> dict:
    """Returns scalar percentile CAGR values (used for dropdown defaults)."""
    returns = np.array(list(annual_pct_data.values()))
    log_r   = np.log(1 + returns / 100)
    mu, sig = log_r.mean(), log_r.std()
    sim     = np.random.normal(mu, sig, (n_sim, n_years))
    cagr    = (np.exp(sim.mean(axis=1)) - 1) * 100
    return {
        "pessimistic":     round(float(np.percentile(cagr, 10)), 2),
        "conservative":    round(float(np.percentile(cagr, 25)), 2),
        "median":          round(float(np.percentile(cagr, 50)), 2),
        "optimistic":      round(float(np.percentile(cagr, 75)), 2),
        "bull":            round(float(np.percentile(cagr, 90)), 2),
        "historical_mean": round(float(returns.mean()), 2),
        "historical_std":  round(float(returns.std()),  2),
    }


# Pre-compute on startup for /api/scenarios endpoint
MC_INVESTMENT = run_monte_carlo_scenarios(SP500_ANNUAL)
MC_INFLATION  = run_monte_carlo_scenarios(CPI_ANNUAL)


# ─────────────────────────────────────────────────────────────────
# 3.  DEBT REPAYMENT  (end-of-month model, cascade on payoff)
# ─────────────────────────────────────────────────────────────────

@dataclass
class Debt:
    name:         str
    balance:      float
    annual_rate:  float    # decimal, e.g. 0.2499
    min_payment:  float
    user_payment: float = 0.0
    extra:        float = 0.0


def _simulate_strategy(debts: List[Debt], strategy: str) -> dict:
    if not debts:
        return {"months": 0, "total_interest": 0.0, "monthly_log": []}

    priority_order = (
        sorted(debts, key=lambda d: d.annual_rate, reverse=True)
        if strategy == "avalanche"
        else sorted(debts, key=lambda d: d.balance)
    )

    working = [
        {
            "name":      d.name,
            "bal":       d.balance,
            "rate":      d.annual_rate,
            "sched_pay": max(d.min_payment, d.user_payment or d.min_payment),
            "extra":     d.extra,
        }
        for d in debts
    ]

    total_extra  = sum(d.extra for d in debts)
    month        = 0
    total_int    = 0.0
    monthly_log  = []
    MAX_MONTHS   = 600

    while any(w["bal"] > 0.005 for w in working) and month < MAX_MONTHS:
        month      += 1
        month_int   = 0.0
        freed_up    = 0.0

        for w in working:
            if w["bal"] <= 0.005:
                w["bal"] = 0.0
                continue
            interest  = w["bal"] * w["rate"] / 12
            w["bal"] += interest
            total_int += interest
            month_int += interest
            paid      = min(w["sched_pay"], w["bal"])
            if w["bal"] - paid < 0.01:
                freed_up += w["sched_pay"] - paid
            w["bal"] = max(0.0, w["bal"] - paid)

        # Cascade freed payments + extra onto priority debt
        pool = total_extra + freed_up
        for sp in priority_order:
            target = next(
                (w for w in working if w["name"] == sp.name and w["bal"] > 0.005),
                None,
            )
            if target:
                apply      = min(pool, target["bal"])
                target["bal"] = max(0.0, target["bal"] - apply)
                break

        monthly_log.append({
            "month":    month,
            "interest": round(month_int, 2),
            "balances": {w["name"]: round(w["bal"], 2) for w in working},
        })

    return {
        "months":         month,
        "total_interest": round(total_int, 2),
        "monthly_log":    monthly_log,
    }


def simulate_debt(debts: List[Debt], strategy: str = "avalanche") -> dict:
    active         = _simulate_strategy(debts, strategy)
    other_strategy = "snowball" if strategy == "avalanche" else "avalanche"
    other          = _simulate_strategy(debts, other_strategy)

    # Baseline: no cascade, min/user payments only
    baseline = [
        {"bal": d.balance, "rate": d.annual_rate,
         "sched": max(d.min_payment, d.user_payment or d.min_payment)}
        for d in debts
    ]
    base_int = 0.0
    bmo      = 0
    while any(b["bal"] > 0.005 for b in baseline) and bmo < 600:
        bmo += 1
        for b in baseline:
            if b["bal"] <= 0.005:
                continue
            interest  = b["bal"] * b["rate"] / 12
            b["bal"] += interest
            base_int += interest
            paid      = min(b["sched"], b["bal"])
            b["bal"]  = max(0.0, b["bal"] - paid)

    return {
        "strategy":            strategy,
        "months":              active["months"],
        "total_interest":      active["total_interest"],
        "interest_saved":      round(base_int - active["total_interest"], 2),
        "monthly_log":         active["monthly_log"],
        "other_strategy":      other_strategy,
        "other_months":        other["months"],
        "other_total_interest": other["total_interest"],
    }


# ─────────────────────────────────────────────────────────────────
# 4.  RETIREMENT PLANNER  (Traditional IRA vs Roth IRA, 2024 limits)
# ─────────────────────────────────────────────────────────────────

@dataclass
class RetirementInputs:
    current_age:         int
    retirement_age:      int
    current_savings:     float
    monthly_savings:     float
    savings_growth_pct:  float   # annual % increase in contribution
    pre_ret_return_pct:  float
    post_ret_return_pct: float
    monthly_expenses:    float
    inflation_pct:       float
    current_income:      float
    current_tax_rate:    float
    retirement_tax_rate: float
    filing_status:       str = "single"


def _ira_limit(age: int) -> float:
    return IRA_LIMIT_OVER50 if age >= 50 else IRA_LIMIT_UNDER50


def _roth_fraction(income: float, status: str) -> float:
    lo, hi = (
        (ROTH_PHASEOUT_MFJ_START, ROTH_PHASEOUT_MFJ_END)
        if status == "mfj"
        else (ROTH_PHASEOUT_SINGLE_START, ROTH_PHASEOUT_SINGLE_END)
    )
    if income <= lo: return 1.0
    if income >= hi: return 0.0
    return 1.0 - (income - lo) / (hi - lo)


def _rmd_divisor(age: int) -> float:
    table = {
        73:26.5,74:25.5,75:24.6,76:23.7,77:22.9,78:22.0,79:21.1,
        80:20.2,81:19.4,82:18.5,83:17.7,84:16.8,85:16.0,86:15.2,
        87:14.4,88:13.7,89:12.9,90:12.2,91:11.5,92:10.8,93:10.1,
        94:9.5, 95:8.9, 96:8.4, 97:7.8, 98:7.3, 99:6.8,100:6.4,
    }
    return table.get(min(age, 100), 6.4)


def simulate_retirement(inp: RetirementInputs) -> dict:
    yrs   = inp.retirement_age - inp.current_age
    if yrs <= 0:
        return {}

    r_pre  = inp.pre_ret_return_pct  / 100
    r_post = inp.post_ret_return_pct / 100
    inf    = inp.inflation_pct       / 100
    sg     = inp.savings_growth_pct  / 100
    mr     = r_pre / 12

    roth_frac   = _roth_fraction(inp.current_income, inp.filing_status)
    ira_limit   = _ira_limit(inp.current_age)
    max_roth    = ira_limit * roth_frac

    port_gen  = inp.current_savings
    port_trad = 0.0
    port_roth = 0.0
    mo_sav    = inp.monthly_savings

    chart_trad: List[float] = []
    chart_roth: List[float] = []
    chart_gen:  List[float] = []
    chart_ages: List[int]   = []

    for y in range(yrs):
        age_now    = inp.current_age + y
        yr_limit   = _ira_limit(age_now)
        yr_rf      = _roth_fraction(inp.current_income, inp.filing_status)
        yr_roth_mo = min(mo_sav, yr_limit * yr_rf / 12)
        yr_trad_mo = min(mo_sav, yr_limit / 12)
        yr_gen_mo  = max(0.0, mo_sav - yr_trad_mo)

        for _ in range(12):
            port_trad = port_trad * (1 + mr) + yr_trad_mo
            port_roth = port_roth * (1 + mr) + yr_roth_mo
            port_gen  = port_gen  * (1 + mr) + yr_gen_mo

        mo_sav *= (1 + sg)
        chart_trad.append(round(port_trad, 0))
        chart_roth.append(round(port_roth, 0))
        chart_gen.append(round(port_gen,   0))
        chart_ages.append(inp.current_age + y + 1)

    total          = port_trad + port_roth + port_gen
    trad_after_tax = port_trad * (1 - inp.retirement_tax_rate)
    roth_after_tax = port_roth
    gen_after_tax  = port_gen  * (1 - inp.retirement_tax_rate * 0.5)
    total_after_tax = trad_after_tax + roth_after_tax + gen_after_tax

    infl_exp       = inp.monthly_expenses * (1 + inf) ** yrs
    annual_wd      = infl_exp * 12
    wd_rate        = annual_wd / total if total else 9.0

    # RMD estimate at age 73
    yrs_to_rmd    = max(0, RMD_START_AGE - inp.retirement_age)
    trad_at_rmd   = port_trad * (1 + r_post) ** yrs_to_rmd
    rmd           = trad_at_rmd / _rmd_divisor(RMD_START_AGE) if trad_at_rmd > 0 else 0.0

    # Depletion estimate
    bal             = total
    depletion_years = 0
    while bal > 0 and depletion_years < 50:
        bal = bal * (1 + r_post) - annual_wd
        depletion_years += 1

    return {
        "years_to_retire":           yrs,
        "portfolio_at_retirement":   round(total, 0),
        "trad_ira_gross":            round(port_trad, 0),
        "trad_ira_after_tax":        round(trad_after_tax, 0),
        "roth_ira_value":            round(port_roth, 0),
        "general_savings":           round(port_gen, 0),
        "total_after_tax":           round(total_after_tax, 0),
        "infl_adj_monthly_exp":      round(infl_exp, 0),
        "annual_withdrawal":         round(annual_wd, 0),
        "withdrawal_rate_pct":       round(wd_rate * 100, 2),
        "rmd_at_73":                 round(rmd, 0),
        "funded_through":            "50+ years" if bal > 0 else f"~{depletion_years} years",
        "target_4pct":               round(annual_wd / 0.04, 0),
        "ira_contribution_limit":    ira_limit,
        "roth_eligible_pct":         round(roth_frac * 100, 1),
        "max_annual_roth":           round(max_roth, 0),
        "chart_ages":                chart_ages,
        "chart_trad":                chart_trad,
        "chart_roth":                chart_roth,
        "chart_general":             chart_gen,
    }


# ─────────────────────────────────────────────────────────────────
# 5.  INVESTMENT GROWTH
# ─────────────────────────────────────────────────────────────────

def calc_investment_growth(initial: float, monthly: float,
                            annual_return_pct: float, years: int) -> dict:
    mr   = annual_return_pct / 100 / 12
    bal  = initial
    contrib = initial
    chart_portfolio = [round(initial, 0)]
    chart_contrib   = [round(initial, 0)]

    for _ in range(years):
        for _ in range(12):
            bal = bal * (1 + mr) + monthly
        contrib += monthly * 12
        chart_portfolio.append(round(bal, 0))
        chart_contrib.append(round(contrib, 0))

    total_contributed = initial + monthly * years * 12
    total_growth      = bal - total_contributed

    return {
        "final_value":         round(bal, 0),
        "total_contributed":   round(total_contributed, 0),
        "total_growth":        round(total_growth, 0),
        "return_on_principal": round(total_growth / total_contributed * 100, 2) if total_contributed else 0,
        "chart_portfolio":     chart_portfolio,
        "chart_contrib":       chart_contrib,
        "years":               list(range(years + 1)),
    }


# ─────────────────────────────────────────────────────────────────
# 6.  LOAN AMORTIZATION
# ─────────────────────────────────────────────────────────────────

def calc_loan_amortization(
    principal: float,
    annual_rate_pct: float,
    years: int,
    extra_payment: float = 0.0,
) -> dict:
    mr = annual_rate_pct / 100 / 12
    n  = years * 12

    if mr > 0:
        min_pay = principal * mr / (1 - (1 + mr) ** (-n))
    else:
        min_pay = principal / n

    payment       = min_pay + extra_payment
    balance       = principal
    total_int     = 0.0
    month         = 0
    schedule      = []
    by_year_bal   = [round(principal, 0)]
    by_year_princ = [0.0]
    by_year_int   = [0.0]
    yr_p = yr_i   = 0.0

    while balance > 0.005 and month < n + 1:
        month   += 1
        interest = balance * mr
        princ    = min(payment - interest, balance)
        balance  = max(0.0, balance - princ)
        total_int += interest
        yr_p += princ
        yr_i += interest

        if month <= 24 or month % 12 == 0 or balance < 0.005:
            schedule.append({
                "month":     month,
                "payment":   round(princ + interest, 2),
                "principal": round(princ, 2),
                "interest":  round(interest, 2),
                "balance":   round(balance, 2),
            })

        if month % 12 == 0:
            by_year_bal.append(round(balance, 0))
            by_year_princ.append(round(yr_p, 0))
            by_year_int.append(round(yr_i, 0))
            yr_p = yr_i = 0.0

    standard_int = min_pay * n - principal if mr > 0 else 0.0

    return {
        "monthly_payment":  round(min_pay, 2),
        "total_interest":   round(total_int, 2),
        "interest_saved":   round(max(0.0, standard_int - total_int), 2),
        "payoff_months":    month,
        "payoff_years":     round(month / 12, 1),
        "schedule":         schedule[:60],
        "chart_balance":    by_year_bal,
        "chart_principal":  by_year_princ,
        "chart_interest":   by_year_int,
        "chart_years":      list(range(len(by_year_bal))),
    }


# ─────────────────────────────────────────────────────────────────
# 7.  CASH FLOW
# ─────────────────────────────────────────────────────────────────

def calc_cash_flow(monthly_income: float, expenses: List[Dict]) -> dict:
    total_exp    = sum(e.get("amount", 0) for e in expenses)
    net          = monthly_income - total_exp
    savings_rate = (net / monthly_income * 100) if monthly_income > 0 else 0

    by_category: Dict[str, float] = {}
    for e in expenses:
        cat = e.get("category", "other")
        by_category[cat] = round(by_category.get(cat, 0) + e.get("amount", 0), 2)

    return {
        "monthly_income": monthly_income,
        "total_expenses": round(total_exp, 2),
        "net_cash_flow":  round(net, 2),
        "savings_rate":   round(savings_rate, 2),
        "by_category":    by_category,
    }
