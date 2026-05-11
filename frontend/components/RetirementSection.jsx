import { useState } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Tooltip, Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { Field, Spinner } from './MonteCarloSection'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const fmt = n => n == null ? '—' : '$' + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 })

const INPUTS = [
  [
    { k: 'current_age',         l: 'Current Age',      s: 'yrs' },
    { k: 'retirement_age',      l: 'Retire At',        s: 'yrs' },
    { k: 'current_savings',     l: 'Current Savings',  p: '$'   },
    { k: 'monthly_savings',     l: 'Monthly Savings',  p: '$'   },
  ],
  [
    { k: 'monthly_expenses',    l: 'Monthly Expenses', p: '$'   },
    { k: 'current_income',      l: 'Annual Income',    p: '$'   },
    { k: 'pre_ret_return_pct',  l: 'Pre-Ret Return',   s: '%'   },
    { k: 'post_ret_return_pct', l: 'Post-Ret Return',  s: '%'   },
  ],
  [
    { k: 'inflation_pct',       l: 'Inflation Rate',   s: '%'   },
    { k: 'savings_growth_pct',  l: 'Contrib. Growth',  s: '%'   },
    { k: 'current_tax_rate',    l: 'Current Tax',      s: '%'   },
    { k: 'retirement_tax_rate', l: 'Ret. Tax',         s: '%'   },
  ],
]

export default function RetirementSection() {
  const [form, setForm] = useState({
    current_age: '30', retirement_age: '65', current_savings: '15000',
    monthly_savings: '800', savings_growth_pct: '3', pre_ret_return_pct: '7',
    post_ret_return_pct: '5', monthly_expenses: '3000', inflation_pct: '2.5',
    current_income: '80000', current_tax_rate: '22', retirement_tax_rate: '15',
    filing_status: 'single',
  })
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function handleSubmit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/api/simulate/retirement`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_age:         parseInt(form.current_age),
          retirement_age:      parseInt(form.retirement_age),
          current_savings:     parseFloat(form.current_savings),
          monthly_savings:     parseFloat(form.monthly_savings),
          savings_growth_pct:  parseFloat(form.savings_growth_pct),
          pre_ret_return_pct:  parseFloat(form.pre_ret_return_pct),
          post_ret_return_pct: parseFloat(form.post_ret_return_pct),
          monthly_expenses:    parseFloat(form.monthly_expenses),
          inflation_pct:       parseFloat(form.inflation_pct),
          current_income:      parseFloat(form.current_income),
          current_tax_rate:    parseFloat(form.current_tax_rate) / 100,
          retirement_tax_rate: parseFloat(form.retirement_tax_rate) / 100,
          filing_status:       form.filing_status,
        }),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Cannot reach API. Make sure the Python backend is running.') }
    finally { setLoading(false) }
  }

  const chartData = result && {
    labels: result.chart_ages,
    datasets: [
      { label: 'Traditional IRA', data: result.chart_trad,    backgroundColor: 'rgba(107,91,149,0.75)', borderRadius: 2, stack: 's' },
      { label: 'Roth IRA',        data: result.chart_roth,    backgroundColor: 'rgba(5,150,105,0.75)',  borderRadius: 2, stack: 's' },
      { label: 'General Savings', data: result.chart_general, backgroundColor: 'rgba(217,119,6,0.70)',  borderRadius: 2, stack: 's' },
    ],
  }

  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 11, family: 'DM Mono' }, color: '#6e6a7c' } },
      tooltip: {
        mode: 'index', intersect: false,
        backgroundColor: 'rgba(26,21,35,.93)', padding: 10, cornerRadius: 8,
        callbacks: { label: c => `  ${c.dataset.label}: ${fmt(c.raw)}` },
      },
    },
    scales: {
      x: { stacked: true, grid: { display: false }, ticks: { font: { size: 10, family: 'DM Mono' }, color: '#9b97a8', maxTicksLimit: 10 } },
      y: { stacked: true, grid: { color: 'rgba(107,91,149,.06)' }, ticks: { callback: v => fmt(v), font: { size: 10, family: 'DM Mono' }, color: '#9b97a8' } },
    },
  }

  return (
    <div className="results-wrap fade-up">
      <div className="results-hdr">
        <div>
          <p className="results-eyebrow">Traditional IRA · Roth IRA · 2024 limits · RMD at 73</p>
          <h1 className="results-title">Your Retirement. <span>Planned.</span></h1>
        </div>
      </div>

      <div className="section-title">🏖️ Retirement Planner</div>

      {/* Inputs */}
      <div className="result-card" style={{ marginBottom: 24 }}>
        <div className="rcard-hdr">
          <div className="rcard-title"><div className="rcard-icon">⚙️</div>Retirement Assumptions</div>
        </div>
        <div className="rcard-body">
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {INPUTS.map((row, ri) => (
              <div key={ri} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
                {row.map(f => <Field key={f.k} label={f.l} prefix={f.p} suffix={f.s} value={form[f.k]} onChange={v => set(f.k, v)} placeholder="" />)}
              </div>
            ))}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 4 }}>
              <span style={{ fontFamily: 'DM Mono', fontSize: 10, color: 'var(--muted2)', textTransform: 'uppercase', letterSpacing: '.08em' }}>Filing Status</span>
              {['single', 'mfj'].map(s => (
                <button key={s} type="button" onClick={() => set('filing_status', s)}
                  className={`toggle-btn${form.filing_status === s ? ' active' : ''}`} style={{ fontSize: 12 }}>
                  {s === 'single' ? 'Single' : 'Married (MFJ)'}
                </button>
              ))}
            </div>
            <div>
              <button type="submit" disabled={loading} className="btn-accent">
                {loading ? <><Spinner /> Calculating…</> : 'Project My Retirement →'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {error && <div className="error-banner" style={{ marginBottom: 20 }}>{error}</div>}

      {result && (
        <>
          <div id="results-container">
            {/* Portfolio summary */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">💼</div>Retirement Portfolio</div>
                <span className={`stat-badge ${result.withdrawal_rate_pct <= 4 ? 'good' : 'warn'}`}>
                  {result.withdrawal_rate_pct <= 4 ? '✓ Safe Rate' : '⚠ High Rate'}
                </span>
              </div>
              <div className="rcard-body">
                <div className="big-metric">
                  <div className="big-metric-value">{fmt(result.portfolio_at_retirement)}</div>
                  <div className="big-metric-label">At Retirement (Age {form.retirement_age})</div>
                </div>
                <div className="metric-row"><span className="metric-label">After-Tax Portfolio</span><span className="metric-value positive">{fmt(result.total_after_tax)}</span></div>
                <div className="metric-row"><span className="metric-label">Annual Withdrawal</span><span className="metric-value">{fmt(result.annual_withdrawal)}</span></div>
                <div className="metric-row">
                  <span className="metric-label">Withdrawal Rate</span>
                  <span className={`metric-value ${result.withdrawal_rate_pct > 4 ? 'negative' : 'positive'}`}>{result.withdrawal_rate_pct}%</span>
                </div>
                <div className="metric-row"><span className="metric-label">Portfolio Sustains</span><span className="metric-value neutral">{result.funded_through}</span></div>
              </div>
            </div>

            {/* IRA breakdown */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📈</div>IRA Breakdown</div>
              </div>
              <div className="rcard-body">
                <div className="metric-row"><span className="metric-label">Traditional IRA (Gross)</span><span className="metric-value">{fmt(result.trad_ira_gross)}</span></div>
                <div className="metric-row"><span className="metric-label">Traditional IRA (After-Tax)</span><span className="metric-value warn">{fmt(result.trad_ira_after_tax)}</span></div>
                <div className="metric-row"><span className="metric-label">Roth IRA (Tax-Free)</span><span className="metric-value positive">{fmt(result.roth_ira_value)}</span></div>
                <div className="metric-row"><span className="metric-label">Roth Eligibility</span><span className="metric-value">{result.roth_eligible_pct}%</span></div>
                <div className="metric-row"><span className="metric-label">Max Annual Roth</span><span className="metric-value">${result.max_annual_roth?.toLocaleString()}</span></div>
              </div>
            </div>

            {/* Sustainability */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📊</div>Sustainability Analysis</div>
              </div>
              <div className="rcard-body">
                <div className="metric-row"><span className="metric-label">RMD at Age 73</span><span className="metric-value warn">{fmt(result.rmd_at_73)}/yr</span></div>
                <div className="metric-row"><span className="metric-label">Infl.-Adj. Monthly Expenses</span><span className="metric-value neutral">{fmt(result.infl_adj_monthly_exp)}</span></div>
                <div className="metric-row"><span className="metric-label">4% Rule Target</span><span className="metric-value">{fmt(result.target_4pct)}</span></div>
                <div className="metric-row"><span className="metric-label">Years to Retire</span><span className="metric-value neutral">{result.years_to_retire}</span></div>
                <div className="metric-row">
                  <span className="metric-label">4% Rate Status</span>
                  <span className={`metric-value ${result.withdrawal_rate_pct <= 4 ? 'positive' : 'negative'}`}>
                    {result.withdrawal_rate_pct <= 4 ? '✓ Within 4% Rule' : '⚠ Exceeds 4% Rule'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Growth chart */}
          <div className="result-card">
            <div className="rcard-hdr">
              <div className="rcard-title"><div className="rcard-icon">📈</div>Portfolio Growth by Account Type</div>
            </div>
            <div className="rcard-body">
              <div className="chart-wrap">
                <Bar data={chartData} options={chartOptions} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
