const NAV = [
  { id: 'montecarlo', label: 'Monte Carlo'       },
  { id: 'investment', label: 'Investment Growth' },
  { id: 'retirement', label: 'Retirement'        },
  { id: 'debt',       label: 'Debt Payoff'       },
  { id: 'loan',       label: 'Loan Calculator'   },
]

export default function Layout({ activeTab, setActiveTab, children }) {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>

      {/* ── Header ── */}
      <header className="app-header">
        <div className="app-header-inner">
          <div className="app-logo">Financial <span>Dashboard</span></div>

          <nav className="nav-tabs">
            {NAV.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`nav-tab${activeTab === item.id ? ' active' : ''}`}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <div className="header-badge">10k sims · 51yr S&P data</div>
        </div>
      </header>

      {/* ── Body ── */}
      <main style={{ flex: 1, paddingTop: 64 }}>
        {children}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <span>FutureFund — Financial Planning Suite</span>
        <span>S&P 500 data 1974–2024 · Not financial advice</span>
      </footer>
    </div>
  )
}
