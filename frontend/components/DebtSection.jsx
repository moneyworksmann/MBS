import { useState } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { Field, Spinner } from './MonteCarloSection'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const fmt = n => n == null ? '—' : '$' + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 })
const mos = m => `${Math.floor(m / 12)}y ${m % 12}m`

const DEBT_COLORS = [
  'rgba(107,91,149,1)', 'rgba(220,53,69,1)',
  'rgba(5,150,105,1)',  'rgba(217,119,6,1)', 'rgba(79,70,229,1)',
]

const DEFAULTS = [
  { name: 'Credit Card',  balance: '4200',  annual_rate: '0.2499', min_payment: '84',  user_payment: '200', extra: '0'   },
  { name: 'Student Loan', balance: '32000', annual_rate: '0.065',  min_payment: '350', user_payment: '500', extra: '100' },
]

export default function DebtSection() {
  const [debts, setDebts]       = useState(DEFAULTS)
  const [strategy, setStrategy] = useState('avalanche')
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  const setDebt = (i, k, v) => setDebts(ds => ds.map((d, idx) => idx === i ? { ...d, [k]: v } : d))
  const addDebt  = () => setDebts(ds => [...ds, { name: `Debt ${ds.length + 1}`, balance: '1000', annual_rate: '0.15', min_payment: '25', user_payment: '0', extra: '0' }])
  const rmDebt   = i => setDebts(ds => ds.filter((_, idx) => idx !== i))

  async function handleSubmit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/api/simulate/debt`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy,
          debts: debts.map(d => ({
            name: d.name, balance: parseFloat(d.balance) || 0,
            annual_rate: parseFloat(d.annual_rate) || 0, min_payment: parseFloat(d.min_payment) || 0,
            user_payment: parseFloat(d.user_payment) || 0, extra: parseFloat(d.extra) || 0,
          })),
        }),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Cannot reach API. Make sure the Python backend is running.') }
    finally { setLoading(false) }
  }

  const chartData = result && (() => {
    const log      = result.monthly_log
    const names    = Object.keys(log[0]?.balances ?? {})
    const sampled  = log.filter((_, i) => i % 3 === 0 || i === log.length - 1)
    return {
      labels: sampled.map(e => `Mo ${e.month}`),
      datasets: names.map((name, i) => ({
        label: name, data: sampled.map(e => e.balances[name] ?? 0),
        borderColor: DEBT_COLORS[i % DEBT_COLORS.length],
        backgroundColor: DEBT_COLORS[i % DEBT_COLORS.length].replace('1)', '0.08)'),
        borderWidth: 2, fill: true, pointRadius: 0, tension: 0.3,
      })),
    }
  })()

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
      x: { grid: { display: false }, ticks: { font: { size: 10, family: 'DM Mono' }, color: '#9b97a8', maxTicksLimit: 10 } },
      y: { grid: { color: 'rgba(107,91,149,.06)' }, ticks: { callback: v => fmt(v), font: { size: 10, family: 'DM Mono' }, color: '#9b97a8' } },
    },
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
  }

  return (
    <div className="results-wrap fade-up">
      <div className="results-hdr">
        <div>
          <p className="results-eyebrow">Avalanche · Snowball · Cascade payments</p>
          <h1 className="results-title">Your Debt. <span>Eliminated.</span></h1>
        </div>
      </div>

      <div className="section-title">💳 Debt Payoff Planner</div>

      {/* Strategy toggle */}
      <div className="result-card" style={{ marginBottom: 16 }}>
        <div className="rcard-hdr">
          <div className="rcard-title"><div className="rcard-icon">🎯</div>Payoff Strategy</div>
        </div>
        <div className="rcard-body">
          <div className="toggle-group">
            {['avalanche', 'snowball'].map(s => (
              <button key={s} onClick={() => setStrategy(s)}
                className={`toggle-btn${strategy === s ? ' active' : ''}`}>
                {s === 'avalanche' ? '🏔 Avalanche — Highest APR First' : '⛄ Snowball — Lowest Balance First'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Debt rows */}
      <form onSubmit={handleSubmit}>
        {debts.map((d, i) => (
          <div key={i} className="debt-row">
            <div className="debt-row-hdr">
              <input
                value={d.name} onChange={e => setDebt(i, 'name', e.target.value)}
                className="debt-row-name"
                style={{ background: 'none', border: 'none', outline: 'none', fontSize: 12, fontWeight: 700, color: 'var(--accent)', fontFamily: 'Syne', cursor: 'text' }}
              />
              {debts.length > 1 && <button type="button" className="rm-btn" onClick={() => rmDebt(i)}>✕</button>}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 10 }}>
              <Field label="Balance"     prefix="$"  value={d.balance}      onChange={v => setDebt(i, 'balance', v)}      placeholder="5,000" />
              <Field label="APR %"       suffix="%"  value={(parseFloat(d.annual_rate)*100).toFixed(2)}
                onChange={v => setDebt(i, 'annual_rate', (parseFloat(v)/100).toString())} placeholder="24.99" />
              <Field label="Min Payment" prefix="$"  value={d.min_payment}  onChange={v => setDebt(i, 'min_payment', v)}  placeholder="50"    />
              <Field label="Your Payment" prefix="$" value={d.user_payment} onChange={v => setDebt(i, 'user_payment', v)} placeholder="200"   />
              <Field label="Extra / mo"  prefix="$"  value={d.extra}        onChange={v => setDebt(i, 'extra', v)}         placeholder="0"     />
            </div>
          </div>
        ))}

        <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
          <button type="button" onClick={addDebt} className="btn-outline">+ Add Debt</button>
          <button type="submit" disabled={loading} className="btn-accent">
            {loading ? <><Spinner /> Calculating…</> : 'Calculate Payoff →'}
          </button>
        </div>
      </form>

      {error && <div className="error-banner" style={{ marginBottom: 20 }}>{error}</div>}

      {result && (
        <>
          <div id="results-container">
            {/* Payoff plan */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">💳</div>Debt Payoff Plan</div>
                <span className="stat-badge good" style={{ textTransform: 'capitalize' }}>{result.strategy}</span>
              </div>
              <div className="rcard-body">
                <div className="big-metric">
                  <div className="big-metric-value">{mos(result.months)}</div>
                  <div className="big-metric-label">Time to Debt-Free</div>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Total Interest Paid</span>
                  <span className="metric-value negative">{fmt(result.total_interest)}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Interest Saved vs Baseline</span>
                  <span className="metric-value positive">{fmt(result.interest_saved)}</span>
                </div>
              </div>
            </div>

            {/* Strategy comparison */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📈</div>Strategy Comparison</div>
              </div>
              <div className="rcard-body">
                <div className="metric-row">
                  <span className="metric-label" style={{ textTransform: 'capitalize' }}>{result.strategy} (selected)</span>
                  <span className="metric-value">{mos(result.months)}</span>
                </div>
                <div className="pbar-wrap" style={{ marginBottom: 16 }}>
                  <div className="pbar"><div className="pbar-fill accent" style={{ width: '100%' }} /></div>
                </div>
                <div className="metric-row">
                  <span className="metric-label" style={{ textTransform: 'capitalize' }}>{result.other_strategy}</span>
                  <span className="metric-value neutral">{mos(result.other_months)}</span>
                </div>
                <div className="pbar-wrap">
                  <div className="pbar">
                    <div className="pbar-fill success" style={{ width: `${Math.min(100, (result.other_months / result.months) * 100)}%` }} />
                  </div>
                </div>
                <div className="metric-row" style={{ marginTop: 16 }}>
                  <span className="metric-label">{result.other_strategy} interest</span>
                  <span className="metric-value warn">{fmt(result.other_total_interest)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Balance chart */}
          <div className="result-card">
            <div className="rcard-hdr">
              <div className="rcard-title"><div className="rcard-icon">📉</div>Debt Balance Over Time</div>
            </div>
            <div className="rcard-body">
              <div className="chart-wrap">
                <Line data={chartData} options={chartOptions} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
