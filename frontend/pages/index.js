import { useState } from 'react'
import Layout from '../components/Layout'
import MonteCarloSection from '../components/MonteCarloSection'
import InvestmentSection from '../components/InvestmentSection'
import RetirementSection from '../components/RetirementSection'
import DebtSection from '../components/DebtSection'
import LoanSection from '../components/LoanSection'

const FEATURES = [
  '💳 Debt Repayment', '🏠 Loan & Mortgage', '💰 Cash Flow',
  '🏖️ Retirement', '📈 Investment Growth', '🏦 Net Worth',
]

export default function Home() {
  const [screen, setScreen]   = useState('landing')   // 'landing' | 'app'
  const [activeTab, setActiveTab] = useState('montecarlo')

  const content = {
    montecarlo: <MonteCarloSection />,
    investment: <InvestmentSection />,
    retirement: <RetirementSection />,
    debt:       <DebtSection />,
    loan:       <LoanSection />,
  }

  /* ── Landing ── */
  if (screen === 'landing') {
    return (
      <div className="landing-wrap fade-up">
        <div className="landing-eyebrow">
          <span className="landing-eyebrow-dot" />
          Personal Finance Suite
        </div>

        <h1 className="landing-title">
          Your Money.<br /><em>Fully Calculated.</em>
        </h1>

        <p className="landing-sub">
          Tell us what you want to plan. We'll show you exactly how the numbers work —
          powered by 10,000 Monte Carlo simulations and 51 years of market history.
        </p>

        <div className="landing-cta-group">
          <button className="btn-accent" style={{ fontSize: 15, padding: '16px 44px' }} onClick={() => setScreen('app')}>
            Get Started →
          </button>
          <span className="landing-note">Free · No account needed</span>
        </div>

        <div className="landing-features">
          {FEATURES.map(f => <span key={f} className="feat-pill">{f}</span>)}
        </div>
      </div>
    )
  }

  /* ── App ── */
  return (
    <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
      {content[activeTab] ?? <MonteCarloSection />}
    </Layout>
  )
}
