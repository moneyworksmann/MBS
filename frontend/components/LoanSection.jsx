import { useState } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Tooltip, Legend, Filler,
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'
import { Field, Spinner } from './MonteCarloSection'

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Tooltip, Legend, Filler)

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const fmt = n => n == null ? '—' : '$' + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 })

const PRESETS = [
  { label: '30yr Mortgage',  principal: '400000', rate: '6.8',  years: '30' },
  { label: '15yr Mortgage',  principal: '400000', rate: '6.2',  years: '15' },
  { label: 'Auto Loan 5yr',  principal: '35000',  rate: '7.5',  years: '5'  },
  { label: 'Student Loan',   principal: '50000',  rate: '5.5',  years: '10' },
]

export default function LoanSection() {
  const [form, setForm] = useState({ principal: '300000', annual_rate_pct: '6.5', years: '30', extra_payment: '0' })
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function handleSubmit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/api/simulate/loan`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          principal:       parseFloat(form.principal),
          annual_rate_pct: parseFloat(form.annual_rate_pct),
          years:           parseInt(form.years),
          extra_payment:   parseFloat(form.extra_payment) || 0,
        }),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Cannot reach API. Make sure the Python backend is running.') }
    finally { setLoading(false) }
  }

  const barData = result && {
    labels: result.chart_years.slice(1).map(y => `Yr ${y}`),
    datasets: [
      { label: 'Principal', data: result.chart_principal.slice(1), backgroundColor: 'rgba(107,91,149,0.75)', borderRadius: 2, stack: 's' },
      { label: 'Interest',  data: result.chart_interest.slice(1),  backgroundColor: 'rgba(220,53,69,0.65)',  borderRadius: 2, stack: 's' },
    ],
  }

  const lineData = result && {
    labels: result.chart_years.map(y => `Yr ${y}`),
    datasets: [{
      label: 'Remaining Balance', data: result.chart_balance,
      borderColor: 'rgba(107,91,149,1)', backgroundColor: 'rgba(107,91,149,0.08)',
      borderWidth: 2.5, fill: true, pointRadius: 0, tension: 0.3,
    }],
  }

  const barOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { usePointStyle: true, padding: 14, font: { size: 11, family: 'DM Mono' }, color: '#6e6a7c' } },
      tooltip: { callbacks: { label: c => `  ${c.dataset.label}: ${fmt(c.raw)}` }, backgroundColor: 'rgba(26,21,35,.93)', padding: 10, cornerRadius: 8 },
    },
    scales: {
      x: { stacked: true, grid: { display: false }, ticks: { font: { size: 10, family: 'DM Mono' }, color: '#9b97a8', maxTicksLimit: 12 } },
      y: { stacked: true, grid: { color: 'rgba(107,91,149,.06)' }, ticks: { callback: v => fmt(v), font: { size: 10, family: 'DM Mono' }, color: '#9b97a8' } },
    },
  }

  const lineOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => `  Balance: ${fmt(c.raw)}` }, backgroundColor: 'rgba(26,21,35,.93)', padding: 10, cornerRadius: 8 },
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { size: 10, family: 'DM Mono' }, color: '#9b97a8', maxTicksLimit: 10 } },
      y: { grid: { color: 'rgba(107,91,149,.06)' }, ticks: { callback: v => fmt(v), font: { size: 10, family: 'DM Mono' }, color: '#9b97a8' } },
    },
  }

  return (
    <div className="results-wrap fade-up">
      <div className="results-hdr">
        <div>
          <p className="results-eyebrow">Amortization · Total interest · Extra payment savings</p>
          <h1 className="results-title">Your Loan. <span>Analysed.</span></h1>
        </div>
      </div>

      <div className="section-title">🏦 Loan Amortization Calculator</div>

      {/* Presets */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
        {PRESETS.map(p => (
          <button key={p.label}
            onClick={() => setForm(f => ({ ...f, principal: p.principal, annual_rate_pct: p.rate, years: p.years }))}
            className="toggle-btn" style={{ fontSize: 11 }}>{p.label}</button>
        ))}
      </div>

      {/* Input card */}
      <div className="result-card" style={{ marginBottom: 24 }}>
        <div className="rcard-hdr">
          <div className="rcard-title"><div className="rcard-icon">📋</div>Loan Details</div>
        </div>
        <div className="rcard-body">
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <Field label="Loan Amount"   prefix="$"   value={form.principal}       onChange={v => set('principal', v)}       placeholder="300,000" />
            <Field label="Interest Rate" suffix="%"   value={form.annual_rate_pct} onChange={v => set('annual_rate_pct', v)} placeholder="6.5"     />
            <Field label="Term"          suffix="yrs" value={form.years}           onChange={v => set('years', v)}           placeholder="30" min="1" max="50" />
            <Field label="Extra / mo"    prefix="$"   value={form.extra_payment}   onChange={v => set('extra_payment', v)}   placeholder="0"       />
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button type="submit" disabled={loading} className="btn-accent">
                {loading ? <><Spinner /> Calculating…</> : 'Calculate →'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {error && <div className="error-banner" style={{ marginBottom: 20 }}>{error}</div>}

      {result && (
        <>
          <div id="results-container">
            {/* Loan details */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📋</div>Loan Summary</div>
              </div>
              <div className="rcard-body">
                <div className="big-metric">
                  <div className="big-metric-value">{fmt(result.monthly_payment)}</div>
                  <div className="big-metric-label">Monthly Payment</div>
                </div>
                <div className="metric-row"><span className="metric-label">Total Interest</span><span className="metric-value negative">{fmt(result.total_interest)}</span></div>
                <div className="metric-row"><span className="metric-label">Interest Saved (Extra Pymts)</span><span className="metric-value positive">{fmt(result.interest_saved)}</span></div>
                <div className="metric-row"><span className="metric-label">Payoff Time</span><span className="metric-value">{result.payoff_years} yrs</span></div>
                {parseFloat(form.extra_payment) > 0 && (
                  <div className="metric-row"><span className="metric-label">Extra per Month</span><span className="metric-value warn">+{fmt(parseFloat(form.extra_payment))}</span></div>
                )}
              </div>
            </div>

            {/* Amortization table */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📊</div>Amortization (first 24 mo.)</div>
              </div>
              <div className="rcard-body" style={{ padding: '14px 22px' }}>
                <div className="amort-scroll">
                  <table className="dt">
                    <thead>
                      <tr>
                        <th>Mo</th><th>Payment</th><th>Principal</th><th>Interest</th><th>Balance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.schedule.slice(0, 24).map(r => (
                        <tr key={r.month}>
                          <td>{r.month}</td>
                          <td>{fmt(r.payment)}</td>
                          <td style={{ color: 'var(--accent)' }}>{fmt(r.principal)}</td>
                          <td style={{ color: 'var(--danger)' }}>{fmt(r.interest)}</td>
                          <td>{fmt(r.balance)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Charts row */}
          <div id="results-container">
            <div className="result-card">
              <div className="rcard-hdr"><div className="rcard-title"><div className="rcard-icon">📉</div>Remaining Balance</div></div>
              <div className="rcard-body"><div className="chart-wrap" style={{ height: 220 }}><Line data={lineData} options={lineOpts} /></div></div>
            </div>
            <div className="result-card">
              <div className="rcard-hdr"><div className="rcard-title"><div className="rcard-icon">📊</div>Annual Principal vs Interest</div></div>
              <div className="rcard-body"><div className="chart-wrap" style={{ height: 220 }}><Bar data={barData} options={barOpts} /></div></div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
