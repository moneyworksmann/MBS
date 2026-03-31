"""
Financial Calculators — Python Backend
======================================================
Covers:
  1. Monte Carlo — S&P 500 investment return scenarios (1974-2024)
  2. Monte Carlo — CPI inflation scenarios (1974-2024, BLS CPI-U)
  3. Debt repayment simulation (end-of-month, user payment, cascade)
  4. Loan amortization
  5. Cash flow analysis
  6. Retirement planner  (Traditional IRA vs Roth IRA, 2024 limits)
  7. Investment growth (FV compound)
  8. Net worth snapshot
"""

import numpy as np  # type: ignore
import copy
import json
from dataclasses import dataclass, field
from typing import List, Optional

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════
# 1.  HISTORICAL DATA  (1974 – 2024, same 51-year window)
# ══════════════════════════════════════════════════════════════════

SP500_ANNUAL = {                                    # total return incl. dividends
    1974:-25.90,1975:37.00,1976:23.60,1977:-7.40,1978:6.40,
    1979:18.20,1980:31.74,1981:-4.92,1982:21.41,1983:22.51,
    1984:6.27,1985:32.16,1986:18.47,1987:5.23,1988:16.81,
    1989:31.49,1990:-3.17,1991:30.55,1992:7.67,1993:9.99,
    1994:1.33,1995:37.43,1996:23.07,1997:33.36,1998:28.34,
    1999:20.89,2000:-9.10,2001:-11.89,2002:-22.10,2003:28.36,
    2004:10.74,2005:4.83,2006:15.61,2007:5.48,2008:-36.55,
    2009:25.94,2010:14.82,2011:2.10,2012:15.89,2013:32.15,
    2014:13.52,2015:1.38,2016:11.74,2017:21.64,2018:-4.23,
    2019:31.21,2020:18.02,2021:28.47,2022:-18.17,2023:26.06,
    2024:23.31,
}

CPI_ANNUAL = {                                      # BLS CPI-U annual % change
    1974:11.06,1975:9.13,1976:5.74,1977:6.45,1978:7.63,
    1979:11.26,1980:13.55,1981:10.33,1982:6.13,1983:3.21,
    1984:4.30,1985:3.56,1986:1.86,1987:3.65,1988:4.14,
    1989:4.82,1990:5.39,1991:4.25,1992:3.03,1993:2.96,
    1994:2.61,1995:2.81,1996:2.93,1997:2.34,1998:1.55,
    1999:2.19,2000:3.37,2001:2.83,2002:1.59,2003:2.27,
    2004:2.68,2005:3.39,2006:3.24,2007:2.85,2008:3.85,
    2009:-0.36,2010:1.64,2011:3.16,2012:2.07,2013:1.46,
    2014:1.62,2015:0.12,2016:1.26,2017:2.13,2018:2.44,
    2019:1.81,2020:1.23,2021:4.70,2022:8.00,2023:4.12,
    2024:2.90,
}

# ── 2024 IRA LIMITS ─────────────────────────────────────────────
IRA_LIMIT_UNDER50   = 7_000    # annual contribution limit
IRA_LIMIT_OVER50    = 8_000    # catch-up limit (age >= 50)
ROTH_PHASEOUT_SINGLE_START = 146_000   # Roth phaseout starts (single filers)
ROTH_PHASEOUT_SINGLE_END   = 161_000
ROTH_PHASEOUT_MFJ_START    = 230_000   # Married filing jointly
ROTH_PHASEOUT_MFJ_END      = 240_000
RMD_START_AGE       = 73      # Required Minimum Distributions start age


# ══════════════════════════════════════════════════════════════════
# 2.  MONTE CARLO SIMULATIONS
# ══════════════════════════════════════════════════════════════════

def run_monte_carlo(annual_pct_data: dict, n_sim: int = 10_000,
                    n_years: int = 30, label: str = "") -> dict:
    """
    Log-normal Monte Carlo on annualised returns.
    Returns percentile-based scenario rates (%).
    """
    returns = np.array(list(annual_pct_data.values()))
    log_r   = np.log(1 + returns / 100)
    mu, sig = log_r.mean(), log_r.std()

    # Simulate n_sim paths of n_years each
    sim = np.random.normal(mu, sig, (n_sim, n_years))

    # Annualised geometric mean per path
    cagr = (np.exp(sim.mean(axis=1)) - 1) * 100

    scenarios = {
        "pessimistic":  round(float(np.percentile(cagr, 10)), 2),
        "conservative": round(float(np.percentile(cagr, 25)), 2),
        "median":       round(float(np.percentile(cagr, 50)), 2),
        "optimistic":   round(float(np.percentile(cagr, 75)), 2),
        "bull":         round(float(np.percentile(cagr, 90)), 2),
        "historical_mean": round(float(returns.mean()), 2),
        "historical_std":  round(float(returns.std()),  2),
        "n_years": n_years,
        "n_sim":   n_sim,
    }
    if label:
        print(f"\n── Monte Carlo: {label} ({n_sim:,} sims × {n_years} yrs) ──")
        for k, v in scenarios.items():
            if isinstance(v, float):
                print(f"   {k:<20}: {v:.2f}%")
    return scenarios


MC_INVESTMENT = run_monte_carlo(SP500_ANNUAL, label="S&P 500 Investment Return")
MC_INFLATION  = run_monte_carlo(CPI_ANNUAL,   label="CPI Inflation")


# ══════════════════════════════════════════════════════════════════
# 3.  DEBT REPAYMENT (end-of-month, user payment, cascade)
# ══════════════════════════════════════════════════════════════════

@dataclass
class Debt:
    name:         str
    balance:      float
    annual_rate:  float          # as decimal, e.g. 0.2499
    min_payment:  float
    user_payment: float = 0.0    # what user actually wants to pay (>= min enforced)
    extra:        float = 0.0    # additional amount towards priority debt

def simulate_debt(debts: List[Debt], strategy: str = "avalanche") -> dict:
    """
    End-of-month model:
      1. Interest accrues on FULL outstanding balance
      2. Payment applied at END of month
      3. Any freed-up money cascades to priority debt
    """
    if not debts:
        return {"months": 0, "total_interest": 0, "interest_saved": 0, "monthly_log": []}

    # Sort order for priority
    if strategy == "avalanche":
        priority_order = sorted(debts, key=lambda d: d.annual_rate, reverse=True)
    else:
        priority_order = sorted(debts, key=lambda d: d.balance)

    working = [{"name": d.name, "bal": d.balance,
                "rate": d.annual_rate,
                "min_pay": d.min_payment,
                "sched_pay": max(d.min_payment, d.user_payment or d.min_payment),
                "extra": d.extra}
               for d in debts]

    total_extra = sum(d.extra for d in debts)
    month = 0
    total_interest = 0
    monthly_log = []
    MAX_MONTHS = 600

    while any(w["bal"] > 0 for w in working) and month < MAX_MONTHS:
        month += 1
        month_interest = 0
        freed_up = 0

        # Step 1 + 2 — interest accrues, then payment
        for w in working:
            if w["bal"] <= 0:
                continue
            interest = w["bal"] * w["rate"] / 12   # interest on full balance
            w["bal"] += interest                     # balance grows first
            total_interest += interest
            month_interest += interest

            paid = min(w["sched_pay"], w["bal"])     # can't overpay
            w["bal"] = max(0.0, w["bal"] - paid)
            if w["bal"] == 0:
                freed_up += w["sched_pay"] - paid    # recapture leftover

        # Step 3 — extra + freed cash → priority debt
        pool = total_extra + freed_up
        for sp in priority_order:
            target = next((w for w in working
                           if w["name"] == sp.name and w["bal"] > 0), None)
            if target:
                apply = min(pool, target["bal"])
                target["bal"] = max(0.0, target["bal"] - apply)
                break

        monthly_log.append({
            "month": month,
            "interest": round(month_interest, 2),
            "balances": {w["name"]: round(w["bal"], 2) for w in working},
        })

    # Baseline: same scheduled payments (user_payment honored) but NO extra, NO cascade
    # This shows what extra payment + cascade actually saves
    baseline = [{"bal": d.balance, "rate": d.annual_rate,
                 "sched": max(d.min_payment, d.user_payment or d.min_payment)} for d in debts]
    base_int = 0
    bmo = 0
    while any(b["bal"] > 0 for b in baseline) and bmo < MAX_MONTHS:
        bmo += 1
        for b in baseline:
            if b["bal"] <= 0:
                continue
            interest = b["bal"] * b["rate"] / 12
            b["bal"] += interest
            base_int += interest
            paid = min(b["sched"], b["bal"])
            b["bal"] = max(0.0, b["bal"] - paid)

    return {
        "months":          month,
        "total_interest":  round(total_interest, 2),
        "interest_saved":  round(base_int - total_interest, 2),
        "monthly_log":     monthly_log[:6],    # first 6 months shown
    }


# ══════════════════════════════════════════════════════════════════
# 4.  RETIREMENT  — Traditional IRA vs Roth IRA
# ══════════════════════════════════════════════════════════════════

@dataclass
class RetirementInputs:
    current_age:        int
    retirement_age:     int
    current_savings:    float
    monthly_savings:    float
    savings_growth_pct: float    # annual % increase in monthly contribution
    pre_ret_return_pct: float    # investment return before retirement
    post_ret_return_pct:float    # investment return after retirement
    monthly_expenses:   float    # current monthly expenses
    inflation_pct:      float    # annual inflation rate
    current_income:     float    # gross annual income (for IRA eligibility)
    current_tax_rate:   float    # current marginal tax rate (decimal)
    retirement_tax_rate:float    # expected marginal tax rate in retirement
    filing_status:      str = "single"   # "single" or "mfj"
    age_at_calc:        int = 0          # set to current_age if 0


def _ira_contribution_limit(age: int) -> float:
    return IRA_LIMIT_OVER50 if age >= 50 else IRA_LIMIT_UNDER50


def _roth_eligible_fraction(income: float, status: str) -> float:
    """Returns fraction of max Roth contribution allowed (0–1)."""
    if status == "mfj":
        lo, hi = ROTH_PHASEOUT_MFJ_START, ROTH_PHASEOUT_MFJ_END
    else:
        lo, hi = ROTH_PHASEOUT_SINGLE_START, ROTH_PHASEOUT_SINGLE_END
    if income <= lo:
        return 1.0
    if income >= hi:
        return 0.0
    return 1.0 - (income - lo) / (hi - lo)


def _rmd_factor(age: int) -> float:
    """IRS Uniform Lifetime Table divisor (simplified)."""
    table = {
        73:26.5,74:25.5,75:24.6,76:23.7,77:22.9,78:22.0,79:21.1,
        80:20.2,81:19.4,82:18.5,83:17.7,84:16.8,85:16.0,86:15.2,
        87:14.4,88:13.7,89:12.9,90:12.2,91:11.5,92:10.8,93:10.1,
        94:9.5,95:8.9,96:8.4,97:7.8,98:7.3,99:6.8,100:6.4,
    }
    return table.get(min(age, 100), 6.4)


def simulate_retirement(inp: RetirementInputs) -> dict:
    years_to_retire = inp.retirement_age - inp.current_age
    if years_to_retire <= 0:
        return {}

    r_pre  = inp.pre_ret_return_pct  / 100
    r_post = inp.post_ret_return_pct / 100
    inf    = inp.inflation_pct        / 100
    sg     = inp.savings_growth_pct   / 100

    # ── IRA eligibility ──────────────────────────────────────────
    annual_ira_limit    = _ira_contribution_limit(inp.current_age)
    roth_fraction       = _roth_eligible_fraction(inp.current_income, inp.filing_status)
    max_roth_contrib    = annual_ira_limit * roth_fraction
    max_trad_contrib    = annual_ira_limit   # Traditional has no income limit for contributions

    # Monthly IRA allocations (capped at limits)
    monthly_roth = min(inp.monthly_savings, max_roth_contrib / 12)
    monthly_trad = min(inp.monthly_savings, max_trad_contrib / 12)

    # ── Accumulation phase (year by year) ────────────────────────
    port_general = inp.current_savings    # taxable/general savings
    port_trad    = 0.0                    # Traditional IRA
    port_roth    = 0.0                    # Roth IRA
    monthly_sav  = inp.monthly_savings
    mr           = r_pre / 12

    trad_yearly  = []    # for chart
    roth_yearly  = []
    gen_yearly   = []
    ages         = []

    for y in range(years_to_retire):
        age_now = inp.current_age + y
        yr_ira_limit = _ira_contribution_limit(age_now)
        yr_roth_frac = _roth_eligible_fraction(inp.current_income, inp.filing_status)
        yr_roth_mo   = min(monthly_sav, yr_ira_limit * yr_roth_frac / 12)
        yr_trad_mo   = min(monthly_sav, yr_ira_limit / 12)
        yr_gen_mo    = max(0, monthly_sav - yr_trad_mo)   # overflow to general

        for m in range(12):
            port_trad    = port_trad    * (1 + mr) + yr_trad_mo
            port_roth    = port_roth    * (1 + mr) + yr_roth_mo
            port_general = port_general * (1 + mr) + yr_gen_mo

        monthly_sav *= (1 + sg)
        trad_yearly.append(round(port_trad, 0))
        roth_yearly.append(round(port_roth, 0))
        gen_yearly.append(round(port_general, 0))
        ages.append(inp.current_age + y + 1)

    total_at_retirement    = port_trad + port_roth + port_general
    trad_after_tax         = port_trad    * (1 - inp.retirement_tax_rate)
    roth_after_tax         = port_roth                         # tax-free
    gen_after_tax          = port_general * (1 - inp.retirement_tax_rate * 0.5)  # partial (cap gains)
    total_after_tax        = trad_after_tax + roth_after_tax + gen_after_tax

    # ── Inflation-adjusted expenses at retirement ─────────────────
    infl_adj_expenses   = inp.monthly_expenses * (1 + inf) ** years_to_retire
    annual_withdrawal   = infl_adj_expenses * 12
    withdrawal_rate     = annual_withdrawal / total_at_retirement if total_at_retirement else 9

    # ── RMD at age 73 (for Traditional IRA) ───────────────────────
    years_to_rmd    = max(0, RMD_START_AGE - inp.retirement_age)
    port_trad_rmd   = port_trad * (1 + r_post) ** years_to_rmd
    rmd_amount      = port_trad_rmd / _rmd_factor(RMD_START_AGE) if port_trad_rmd > 0 else 0

    # ── Depletion estimate ─────────────────────────────────────────
    bal = total_at_retirement
    depletion_years = 0
    while bal > 0 and depletion_years < 50:
        bal = bal * (1 + r_post) - annual_withdrawal
        depletion_years += 1

    # ── 4% rule target ─────────────────────────────────────────────
    target_4pct = annual_withdrawal / 0.04

    return {
        "years_to_retire":       years_to_retire,
        "portfolio_at_retirement": round(total_at_retirement, 0),
        "trad_ira_gross":        round(port_trad, 0),
        "trad_ira_after_tax":    round(trad_after_tax, 0),
        "roth_ira_value":        round(port_roth, 0),
        "general_savings":       round(port_general, 0),
        "total_after_tax":       round(total_after_tax, 0),
        "infl_adj_monthly_exp":  round(infl_adj_expenses, 0),
        "annual_withdrawal":     round(annual_withdrawal, 0),
        "withdrawal_rate_pct":   round(withdrawal_rate * 100, 2),
        "rmd_at_73":             round(rmd_amount, 0),
        "depletion_years":       depletion_years,
        "funded_through":        "50+ years" if bal > 0 else f"~{depletion_years} years",
        "target_4pct":           round(target_4pct, 0),
        "ira_contribution_limit":annual_ira_limit,
        "roth_eligible_pct":     round(roth_fraction * 100, 1),
        "max_annual_roth":       round(max_roth_contrib, 0),
        "chart_ages":            ages,
        "chart_trad":            trad_yearly,
        "chart_roth":            roth_yearly,
        "chart_general":         gen_yearly,
    }


# ══════════════════════════════════════════════════════════════════
# 5.  INVESTMENT GROWTH (FV compound with monthly contributions)
# ══════════════════════════════════════════════════════════════════

def calc_investment_growth(initial: float, monthly: float,
                            annual_return_pct: float, years: int) -> dict:
    mr = annual_return_pct / 100 / 12
    n  = years * 12
    bal = initial
    portfolio_by_year = [initial]
    contrib_by_year   = [initial]
    contrib = initial

    for y in range(years):
        for _ in range(12):
            bal = bal * (1 + mr) + monthly
        contrib += monthly * 12
        portfolio_by_year.append(round(bal, 0))
        contrib_by_year.append(round(contrib, 0))

    total_contributed = initial + monthly * n
    total_growth      = bal - total_contributed
    return {
        "final_value":       round(bal, 0),
        "total_contributed": round(total_contributed, 0),
        "total_growth":      round(total_growth, 0),
        "return_on_principal": round(total_growth / total_contributed * 100, 2)
                               if total_contributed else 0,
        "chart_portfolio": portfolio_by_year,
        "chart_contrib":   contrib_by_year,
    }


# ══════════════════════════════════════════════════════════════════
# 6.  DEMO RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DEMO — FINANCIAL CALCULATORS")
    print("="*60)

    # ── Debt repayment ──────────────────────────────────────────
    debts = [
        Debt("Credit Card 1", 4200,  0.2499, min_payment=84,  user_payment=200, extra=0),
        Debt("Credit Card 2", 2800,  0.1999, min_payment=56,  user_payment=150, extra=0),
        Debt("Student Loan",  32000, 0.065,  min_payment=350, user_payment=500, extra=100),
    ]
    dr = simulate_debt(debts, "avalanche")
    print(f"\n── Debt Repayment (end-of-month model) ──")
    print(f"  Payoff time    : {dr['months']//12}y {dr['months']%12}m")
    print(f"  Total interest : ${dr['total_interest']:,.2f}")
    print(f"  Interest saved : ${dr['interest_saved']:,.2f}")
    print(f"  First 3 months:")
    for m in dr["monthly_log"][:3]:
        bals = " | ".join(f"{k}: ${v:,.0f}" for k,v in m["balances"].items())
        print(f"    Mo {m['month']:>2}: interest=${m['interest']:,.2f}  |  {bals}")

    # ── Retirement ──────────────────────────────────────────────
    ret_inp = RetirementInputs(
        current_age=30, retirement_age=65,
        current_savings=15000, monthly_savings=800,
        savings_growth_pct=3, pre_ret_return_pct=MC_INVESTMENT["median"],
        post_ret_return_pct=4, monthly_expenses=3000,
        inflation_pct=MC_INFLATION["median"],
        current_income=80000, current_tax_rate=0.22,
        retirement_tax_rate=0.15, filing_status="single",
    )
    ret = simulate_retirement(ret_inp)
    print(f"\n── Retirement (Trad IRA vs Roth IRA) ──")
    print(f"  Portfolio at retirement     : ${ret['portfolio_at_retirement']:>12,.0f}")
    print(f"  Traditional IRA (gross)     : ${ret['trad_ira_gross']:>12,.0f}")
    print(f"  Traditional IRA (after-tax) : ${ret['trad_ira_after_tax']:>12,.0f}")
    print(f"  Roth IRA value              : ${ret['roth_ira_value']:>12,.0f}")
    print(f"  Total after-tax             : ${ret['total_after_tax']:>12,.0f}")
    print(f"  Withdrawal rate             : {ret['withdrawal_rate_pct']:.2f}%")
    print(f"  RMD at age 73               : ${ret['rmd_at_73']:>12,.0f}/yr")
    print(f"  Portfolio sustains          : {ret['funded_through']}")
    print(f"  Roth eligibility            : {ret['roth_eligible_pct']}% of limit")
    print(f"  Annual IRA limit            : ${ret['ira_contribution_limit']:,}")
    print(f"  Max annual Roth contrib     : ${ret['max_annual_roth']:,}")
    print(f"  Inflation used (MC median)  : {MC_INFLATION['median']}%")
    print(f"  Return used (MC median)     : {MC_INVESTMENT['median']}%")

    # ── Investment growth ────────────────────────────────────────
    inv = calc_investment_growth(10000, 500, MC_INVESTMENT["median"], 30)
    print(f"\n── Investment Growth ({MC_INVESTMENT['median']}% return, 30 yrs) ──")
    print(f"  Final value       : ${inv['final_value']:>12,.0f}")
    print(f"  Total contributed : ${inv['total_contributed']:>12,.0f}")
    print(f"  Total gains       : ${inv['total_growth']:>12,.0f}")

    # ── Monte Carlo summary ──────────────────────────────────────
    print(f"\n── MC Investment Return Scenarios (use in HTML dropdowns) ──")
    for k,v in MC_INVESTMENT.items():
        if isinstance(v, float): print(f"  {k:<20}: {v}%")

    print(f"\n── MC Inflation Scenarios ──")
    for k,v in MC_INFLATION.items():
        if isinstance(v, float): print(f"  {k:<20}: {v}%")
