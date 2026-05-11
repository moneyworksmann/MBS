"""FutureFund API — FastAPI backend."""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from calculators import (
    run_monte_carlo_paths,
    simulate_debt, Debt,
    simulate_retirement, RetirementInputs,
    calc_investment_growth,
    calc_loan_amortization,
    calc_cash_flow,
    SP500_ANNUAL,
    MC_INVESTMENT, MC_INFLATION,
)

app = FastAPI(
    title="FutureFund API",
    description="Monte Carlo simulations, debt payoff, retirement planning, and more.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ────────────────────────────────────────────────

class MonteCarloRequest(BaseModel):
    initial_investment:   float = Field(10_000, ge=0)
    monthly_contribution: float = Field(500,    ge=0)
    years:                int   = Field(30,     ge=1, le=50)
    n_simulations:        int   = Field(10_000, ge=100, le=50_000)


class DebtItem(BaseModel):
    name:         str
    balance:      float = Field(ge=0)
    annual_rate:  float = Field(ge=0, le=1)  # decimal  e.g. 0.2499
    min_payment:  float = Field(ge=0)
    user_payment: float = Field(0.0, ge=0)
    extra:        float = Field(0.0, ge=0)


class DebtRequest(BaseModel):
    debts:    List[DebtItem]
    strategy: str = "avalanche"


class RetirementRequest(BaseModel):
    current_age:         int   = Field(30,   ge=18, le=100)
    retirement_age:      int   = Field(65,   ge=18, le=100)
    current_savings:     float = Field(0,    ge=0)
    monthly_savings:     float = Field(500,  ge=0)
    savings_growth_pct:  float = Field(3.0)
    pre_ret_return_pct:  float = Field(7.0)
    post_ret_return_pct: float = Field(5.0)
    monthly_expenses:    float = Field(3_000, ge=0)
    inflation_pct:       float = Field(2.5)
    current_income:      float = Field(80_000, ge=0)
    current_tax_rate:    float = Field(0.22)
    retirement_tax_rate: float = Field(0.15)
    filing_status:       str   = "single"


class InvestmentGrowthRequest(BaseModel):
    initial:           float = Field(10_000, ge=0)
    monthly:           float = Field(500,    ge=0)
    annual_return_pct: float = Field(7.0)
    years:             int   = Field(30, ge=1, le=50)


class LoanRequest(BaseModel):
    principal:       float = Field(ge=0)
    annual_rate_pct: float = Field(ge=0, le=100)
    years:           int   = Field(ge=1, le=50)
    extra_payment:   float = Field(0.0, ge=0)


class ExpenseItem(BaseModel):
    name:     str
    amount:   float
    category: str = "other"


class CashFlowRequest(BaseModel):
    monthly_income: float
    expenses:       List[ExpenseItem]


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/scenarios")
def get_scenarios():
    """Pre-computed MC scenarios used to populate form dropdowns."""
    return {"investment": MC_INVESTMENT, "inflation": MC_INFLATION}


@app.post("/api/simulate/monte-carlo")
def monte_carlo(req: MonteCarloRequest):
    return run_monte_carlo_paths(
        initial=req.initial_investment,
        monthly=req.monthly_contribution,
        n_sim=req.n_simulations,
        n_years=req.years,
        annual_pct_data=SP500_ANNUAL,
    )


@app.post("/api/simulate/debt")
def debt_payoff(req: DebtRequest):
    if req.strategy not in ("avalanche", "snowball"):
        raise HTTPException(400, "strategy must be 'avalanche' or 'snowball'")
    debts = [
        Debt(
            name=d.name,
            balance=d.balance,
            annual_rate=d.annual_rate,
            min_payment=d.min_payment,
            user_payment=d.user_payment,
            extra=d.extra,
        )
        for d in req.debts
    ]
    return simulate_debt(debts, req.strategy)


@app.post("/api/simulate/retirement")
def retirement(req: RetirementRequest):
    if req.retirement_age <= req.current_age:
        raise HTTPException(400, "retirement_age must be greater than current_age")
    inp = RetirementInputs(
        current_age=req.current_age,
        retirement_age=req.retirement_age,
        current_savings=req.current_savings,
        monthly_savings=req.monthly_savings,
        savings_growth_pct=req.savings_growth_pct,
        pre_ret_return_pct=req.pre_ret_return_pct,
        post_ret_return_pct=req.post_ret_return_pct,
        monthly_expenses=req.monthly_expenses,
        inflation_pct=req.inflation_pct,
        current_income=req.current_income,
        current_tax_rate=req.current_tax_rate,
        retirement_tax_rate=req.retirement_tax_rate,
        filing_status=req.filing_status,
    )
    return simulate_retirement(inp)


@app.post("/api/simulate/investment-growth")
def investment_growth(req: InvestmentGrowthRequest):
    return calc_investment_growth(req.initial, req.monthly, req.annual_return_pct, req.years)


@app.post("/api/simulate/loan")
def loan(req: LoanRequest):
    return calc_loan_amortization(req.principal, req.annual_rate_pct, req.years, req.extra_payment)


@app.post("/api/simulate/cash-flow")
def cash_flow(req: CashFlowRequest):
    expenses = [{"name": e.name, "amount": e.amount, "category": e.category}
                for e in req.expenses]
    return calc_cash_flow(req.monthly_income, expenses)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
