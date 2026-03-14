import { motion } from 'framer-motion'
import DonutChart from './DonutChart'

export default function PortfolioSummary({ data }) {
  const totalCost = data.total_cost || 0
  const maxPayout = data.total_max_profit || data.max_payout || 0
  const totalExposure = data.total_exposure || 0
  const positions = data.positions || []
  const hedgedCount = positions.length
  const riskCount = data.risks?.length || 0
  const coverageRatio = totalCost > 0 ? (maxPayout / totalCost).toFixed(1) : '—'
  const exposurePct = totalExposure > 0 ? ((totalCost / totalExposure) * 100).toFixed(1) : null

  return (
    <motion.div
      className="verdict-hero"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="verdict-hero-main">
        <div className="verdict-hero-left">
          <div className="verdict-hero-headline">
            {riskCount > 0
              ? `${riskCount} risks identified, ${hedgedCount} hedged`
              : `${hedgedCount} positions recommended`}
          </div>
          {data.budget_rationale && (
            <div className="verdict-hero-rationale">
              {data.budget_rationale}
            </div>
          )}
          <div className="verdict-hero-stats">
            <div className="verdict-hero-stat">
              <span className="verdict-hero-stat-label">Recommended Premium</span>
              <span className="verdict-hero-stat-value">
                ${totalCost.toLocaleString()}
                {exposurePct && <span className="verdict-hero-pct"> ({exposurePct}% of exposure)</span>}
              </span>
            </div>
            <span className="verdict-hero-divider" />
            <div className="verdict-hero-stat">
              <span className="verdict-hero-stat-label">Max Protection</span>
              <span className="verdict-hero-stat-value">${maxPayout.toLocaleString()}</span>
            </div>
          </div>
        </div>
        <div className="verdict-hero-right">
          <div className="verdict-hero-ratio">
            <span className="verdict-hero-ratio-value">{coverageRatio}x</span>
            <span className="verdict-hero-ratio-label">Coverage</span>
          </div>
          <DonutChart positions={positions} size={48} />
        </div>
      </div>
    </motion.div>
  )
}
