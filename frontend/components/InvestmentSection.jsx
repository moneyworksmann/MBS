import { useState } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { Field, Spinner } from './MonteCarloSection'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const fmt = n => n == null ? '—'
  : n >= 1_000_000 ? '$' + (n / 1_000_000).toFixed(2) + 'M'
  : n >= 1_000     ? '$' + (n / 1_000).toFixed(1) + 'K'
  : '$' + Number(n).toLocaleString()

const PRESETS = [
  { label: 'Conservative 5%',  pct: '5'    },
  { label: 'Moderate 7%',      pct: '7'    },
  { label: 'Aggressive 10%',   pct: '10'   },
  { label: 'S&P Hist. 10.5%',  pct: '10.5' },
]

export default function InvestmentSection() {
  const [form, setForm]     = useState({ initial: '10000', monthly: '500', annual_return_pct: '7', years: '30' })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function handleSubmit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/api/simulate/investment-growth`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial:           parseFloat(form.initial)           || 0,
          monthly:           parseFloat(form.monthly)           || 0,
          annual_return_pct: parseFloat(form.annual_return_pct) || 0,
          years:             parseInt(form.years)               || 30,
        }),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch { setError('Cannot reach API. Make sure the Python backend is running.') }
    finally { setLoading(false) }
  }

  const chartData = result && {
    labels: result.years.map(y => `Yr ${y}`),
    datasets: [
      {
        label: 'Portfolio Value', data: result.chart_portfolio,
        borderColor: 'rgba(107,91,149,1)', backgroundColor: 'rgba(107,91,149,0.10)',
        borderWidth: 2.5, fill: true, pointRadius: 0, tension: 0.4,
      },
      {
        label: 'Total Contributed', data: result.chart_contrib,
        borderColor: 'rgba(217,119,6,0.8)', backgroundColor: 'rgba(217,119,6,0.07)',
        borderWidth: 2, borderDash: [5, 4], fill: true, pointRadius: 0, tension: 0.4,
      },
    ],
  }

  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { usePointStyle: true, padding: 18, font: { size: 11, family: 'DM Mono' }, color: '#6e6a7c' } },
      tooltip: {
        mode: 'index', intersect: false,
        backgroundColor: 'rgba(26,21,35,.93)', padding: 12, cornerRadius: 8,
        callbacks: { label: c => `  ${c.dataset.label}: ${fmt(c.raw)}` },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { size: 10, family: 'DM Mono' }, color: '#9b97a8', maxTicksLimit: 10 } },
      y: { grid: { color: 'rgba(107,91,149,.06)' }, ticks: { callback: v => fmt(v), font: { size: 10, family: 'DM Mono' }, color: '#9b97a8' } },
    },
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
    animation: { duration: 600 },
  }

  return (
    <div className="results-wrap fade-up">
      <div className="results-hdr">
        <div>
          <p className="results-eyebrow">Compound interest · Monthly contributions · Future value</p>
          <h1 className="results-title">Your Investments. <span>Compounded.</span></h1>
        </div>
      </div>

      <div className="section-title">📈 Investment Growth Calculator</div>

      {/* Scenario presets */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
        {PRESETS.map(p => (
          <button key={p.label} onClick={() => set('annual_return_pct', p.pct)}
            className={`toggle-btn${form.annual_return_pct === p.pct ? ' active' : ''}`} style={{ fontSize: 11 }}>
            {p.label}
          </button>
        ))}
      </div>

      {/* Input card */}
      <div className="result-card" style={{ marginBottom: 24 }}>
        <div className="rcard-hdr">
          <div className="rcard-title"><div className="rcard-icon">⚙️</div>Growth Parameters</div>
        </div>
        <div className="rcard-body">
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <Field label="Initial Amount"   prefix="$"   value={form.initial}           onChange={v => set('initial', v)}           placeholder="10,000" />
            <Field label="Monthly Addition" prefix="$"   value={form.monthly}           onChange={v => set('monthly', v)}           placeholder="500"    />
            <Field label="Annual Return"    suffix="%"   value={form.annual_return_pct} onChange={v => set('annual_return_pct', v)} placeholder="7"      />
            <Field label="Years"            suffix="yrs" value={form.years}             onChange={v => set('years', v)}             placeholder="30" min="1" max="50" />
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
            {/* Growth projection */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">💹</div>Growth Projection</div>
              </div>
              <div className="rcard-body">
                <div className="big-metric">
                  <div className="big-metric-value">{fmt(result.final_value)}</div>
                  <div className="big-metric-label">Final Value After {form.years} Years</div>
                </div>
                <div className="metric-row"><span className="metric-label">Total Contributed</span><span className="metric-value warn">{fmt(result.total_contributed)}</span></div>
                <div className="metric-row"><span className="metric-label">Investment Gains</span><span className="metric-value positive">{fmt(result.total_growth)}</span></div>
              </div>
            </div>

            {/* Return analysis */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">🎯</div>Return Analysis</div>
              </div>
              <div className="rcard-body">
                <div className="big-metric">
                  <div className="big-metric-value">{result.return_on_principal}%</div>
                  <div className="big-metric-label">Return on Principal</div>
                </div>
                <div className="metric-row"><span className="metric-label">Initial Investment</span><span className="metric-value neutral">{fmt(parseFloat(form.initial))}</span></div>
                <div className="metric-row"><span className="metric-label">Monthly Contribution</span><span className="metric-value neutral">{fmt(parseFloat(form.monthly))}/mo</span></div>
                <div className="metric-row"><span className="metric-label">Annual Return Rate</span><span className="metric-value">{form.annual_return_pct}%</span></div>
                <div style={{ marginTop: 12 }}>
                  <div className="pbar-wrap">
                    <div className="pbar-label"><span>Gains vs Contributions</span><span>{Math.round(result.total_growth / (result.final_value || 1) * 100)}%</span></div>
                    <div className="pbar"><div className="pbar-fill accent" style={{ width: `${Math.round(result.total_growth / (result.final_value || 1) * 100)}%` }} /></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="result-card">
            <div className="rcard-hdr">
              <div className="rcard-title"><div className="rcard-icon">📈</div>Portfolio Value vs Total Contributed</div>
              <span style={{ fontFamily: 'DM Mono', fontSize: 10, color: 'var(--muted2)' }}>Gap = compound growth</span>
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
