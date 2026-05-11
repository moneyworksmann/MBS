import { useState } from 'react'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

function fmt(n) {
  if (n == null) return '—'
  if (n >= 1_000_000) return '$' + (n / 1_000_000).toFixed(2) + 'M'
  if (n >= 1_000)     return '$' + (n / 1_000).toFixed(1) + 'K'
  return '$' + Number(n).toLocaleString()
}

// p5→p95 fan chart, fill:'-1' stacks colour bands
const PCTS = [
  { key: 'p5',  label: '5th  (Bear)',   border: 'rgba(220,53,69,0.7)',   bg: 'rgba(220,53,69,0.05)',   w: 1.5, dash: true  },
  { key: 'p25', label: '25th',          border: 'rgba(217,119,6,0.7)',   bg: 'rgba(217,119,6,0.07)',   w: 1.5, dash: false },
  { key: 'p50', label: '50th (Median)', border: 'rgba(107,91,149,1)',    bg: 'rgba(107,91,149,0.10)',  w: 3.0, dash: false },
  { key: 'p75', label: '75th',          border: 'rgba(5,150,105,0.7)',   bg: 'rgba(5,150,105,0.07)',   w: 1.5, dash: false },
  { key: 'p95', label: '95th (Bull)',   border: 'rgba(5,150,105,0.9)',   bg: 'rgba(5,150,105,0.05)',   w: 1.5, dash: false },
]

export function Spinner() {
  return (
    <svg className="animate-spin" style={{ width: 16, height: 16 }} fill="none" viewBox="0 0 24 24">
      <circle style={{ opacity: .25 }} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path  style={{ opacity: .75 }} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

export function Field({ label, prefix, suffix, value, onChange, placeholder, min, max }) {
  return (
    <div className="field" style={{ flex: 1, minWidth: 0 }}>
      <label>{label}</label>
      <div style={{ position: 'relative' }}>
        {prefix && (
          <span style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted2)', fontSize: 14, pointerEvents: 'none' }}>
            {prefix}
          </span>
        )}
        <input
          type="number" value={value} placeholder={placeholder} min={min} max={max}
          onChange={e => onChange(e.target.value)}
          style={{ paddingLeft: prefix ? 28 : 14, paddingRight: suffix ? 40 : 14 }}
        />
        {suffix && (
          <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted2)', fontSize: 13, pointerEvents: 'none' }}>
            {suffix}
          </span>
        )}
      </div>
    </div>
  )
}

export default function MonteCarloSection() {
  const [form, setForm]     = useState({ initial: '10000', monthly: '500', years: '30' })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/api/simulate/monte-carlo`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_investment:   parseFloat(form.initial)  || 0,
          monthly_contribution: parseFloat(form.monthly)  || 0,
          years:                parseInt(form.years)      || 30,
          n_simulations:        10_000,
        }),
      })
      if (!res.ok) throw new Error()
      setResult(await res.json())
    } catch {
      setError(`Cannot reach API at ${API}. Run: cd backend && uvicorn main:app --reload`)
    } finally { setLoading(false) }
  }

  const chartData = result && {
    labels: result.years.map(y => y % 5 === 0 ? `Yr ${y}` : ''),
    datasets: PCTS.map((p, i) => ({
      label: p.label, data: result.paths[p.key],
      borderColor: p.border, backgroundColor: p.bg,
      borderWidth: p.w, borderDash: p.dash ? [5, 4] : undefined,
      fill: i === 0 ? false : '-1', pointRadius: 0, tension: 0.4,
    })),
  }

  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { usePointStyle: true, padding: 18, font: { size: 11, family: 'DM Mono' }, color: '#6e6a7c' } },
      tooltip: {
        mode: 'index', intersect: false,
        backgroundColor: 'rgba(26,21,35,.93)', padding: 12, cornerRadius: 8,
        callbacks: { title: ([c]) => `Year ${c.dataIndex}`, label: c => `  ${c.dataset.label}: ${fmt(c.raw)}` },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { size: 10, family: 'DM Mono' }, color: '#9b97a8', maxTicksLimit: 10 } },
      y: { grid: { color: 'rgba(107,91,149,.06)' }, ticks: { callback: v => fmt(v), font: { size: 10, family: 'DM Mono' }, color: '#9b97a8' } },
    },
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
    animation: { duration: 700 },
  }

  return (
    <div className="results-wrap fade-up">
      {/* Page header */}
      <div className="results-hdr">
        <div>
          <p className="results-eyebrow">S&P 500 · 10,000 simulations · 51 years of data</p>
          <h1 className="results-title">Your Money. <span>Visualized.</span></h1>
        </div>
      </div>

      <div className="section-title">📈 Monte Carlo Simulation</div>

      {/* Input card */}
      <div className="result-card" style={{ marginBottom: 24 }}>
        <div className="rcard-hdr">
          <div className="rcard-title"><div className="rcard-icon">⚙️</div>Simulation Parameters</div>
        </div>
        <div className="rcard-body">
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <Field label="Initial Investment"   prefix="$"   value={form.initial} onChange={v => set('initial', v)}  placeholder="10,000" />
            <Field label="Monthly Contribution" prefix="$"   value={form.monthly} onChange={v => set('monthly', v)}  placeholder="500"    />
            <Field label="Time Horizon"         suffix="yrs" value={form.years}   onChange={v => set('years', v)}    placeholder="30" min="1" max="50" />
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button type="submit" disabled={loading} className="btn-accent" style={{ whiteSpace: 'nowrap' }}>
                {loading ? <><Spinner /> Simulating…</> : 'Run Simulation →'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {error && <div className="error-banner" style={{ marginBottom: 20 }}>{error}</div>}

      {/* Results */}
      {result && (
        <>
          <div id="results-container">
            {/* Percentile outcomes */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📊</div>Final Portfolio Value</div>
                <span className="stat-badge good">{result.n_sim.toLocaleString()} sims</span>
              </div>
              <div className="rcard-body">
                <div className="big-metric">
                  <div className="big-metric-value">{fmt(result.final_stats.p50)}</div>
                  <div className="big-metric-label">Median After {form.years} Years</div>
                </div>
                {[
                  { label: '5th  Percentile (Bear)',  key: 'p5',  cls: 'negative' },
                  { label: '25th Percentile',          key: 'p25', cls: 'warn'     },
                  { label: '50th Percentile (Median)', key: 'p50', cls: ''         },
                  { label: '75th Percentile',          key: 'p75', cls: 'positive' },
                  { label: '95th Percentile (Bull)',   key: 'p95', cls: 'positive' },
                ].map(r => (
                  <div key={r.key} className="metric-row">
                    <span className="metric-label">{r.label}</span>
                    <span className={`metric-value ${r.cls}`}>{fmt(result.final_stats[r.key])}</span>
                  </div>
                ))}
                <div className="metric-row">
                  <span className="metric-label">Total Contributed</span>
                  <span className="metric-value neutral">{fmt(result.total_contributed)}</span>
                </div>
              </div>
            </div>

            {/* CAGR scenarios */}
            <div className="result-card">
              <div className="rcard-hdr">
                <div className="rcard-title"><div className="rcard-icon">📉</div>Return Scenarios (CAGR)</div>
              </div>
              <div className="rcard-body">
                {Object.entries(result.scenarios).map(([k, v]) => (
                  <div key={k} className="metric-row">
                    <span className="metric-label" style={{ textTransform: 'capitalize' }}>{k}</span>
                    <span className="metric-value">{v}%</span>
                  </div>
                ))}
                <div style={{ marginTop: 16 }}>
                  <div className="pbar-wrap">
                    <div className="pbar-label"><span>Historical Mean</span><span>{result.hist_mean}%</span></div>
                    <div className="pbar"><div className="pbar-fill accent" style={{ width: `${Math.min(100, (result.hist_mean / 30) * 100)}%` }} /></div>
                  </div>
                  <div className="pbar-wrap">
                    <div className="pbar-label"><span>Historical Std Dev</span><span>±{result.hist_std}%</span></div>
                    <div className="pbar"><div className="pbar-fill success" style={{ width: `${Math.min(100, (result.hist_std / 40) * 100)}%` }} /></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Fan chart */}
          <div className="result-card">
            <div className="rcard-hdr">
              <div className="rcard-title"><div className="rcard-icon">📈</div>Portfolio Growth Fan Chart</div>
              <span style={{ fontFamily: 'DM Mono', fontSize: 10, color: 'var(--muted2)' }}>
                {fmt(parseFloat(form.initial))} + {fmt(parseFloat(form.monthly))}/mo
              </span>
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
