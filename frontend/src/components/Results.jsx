import { useState } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, Info, ChevronDown, ChevronUp, ArrowLeft, ExternalLink } from 'lucide-react'
import PortfolioSummary from './PortfolioSummary'
import Sparkline from './Sparkline'

function PositionCard({ position, index, totalCost }) {
  const [expanded, setExpanded] = useState(false)
  const side = position.side?.toUpperCase()
  const price = position.current_price ?? position.price
  const allocation = position.allocation || position.cost || 0
  const payout = price > 0 ? Math.round(allocation / price) : 0
  const returnMultiple = price > 0 ? ((1 / price) - 1).toFixed(1) : '—'
  const portfolioPct = totalCost > 0 ? Math.round((allocation / totalCost) * 100) : 0
  const confidence = position.confidence || 0
  const confidenceLabel = confidence >= 0.8 ? 'High' : confidence >= 0.5 ? 'Med' : 'Low'
  const confidenceClass = confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'med' : 'low'

  return (
    <motion.div
      className="position-card"
      onClick={() => setExpanded(!expanded)}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
    >
      <div className="position-card-top">
        <div className="position-card-left">
          <div className="position-card-title-row">
            <div className="position-card-title">{position.contract_title}</div>
            <span className={`confidence-badge ${confidenceClass}`}>{confidenceLabel}</span>
          </div>
          <div className="position-card-subtitle">
            {portfolioPct}% of portfolio — hedges {position.risk?.event || 'identified risk'}
          </div>
        </div>
        <div className="position-card-right">
          <div className="position-card-glance">
            <span className="position-card-cost-label">Return</span>
            <span className="position-card-multiple">{returnMultiple}x</span>
            <span className="position-card-cost">{portfolioPct}% of premium</span>
          </div>
        </div>
      </div>

      {expanded && (
        <div className="position-card-detail">
          <div className="detail-top-row">
            <span className={`position-side-badge ${side === 'YES' ? 'yes' : 'no'}`}>
              {side}
            </span>
            <Sparkline seed={position.contract_title} color={side === 'YES' ? 'var(--green)' : 'var(--red)'} />
          </div>
          <div className="detail-grid">
            <div>
              <div className="detail-item-label">Entry Price</div>
              <div className="detail-item-value">${price.toFixed(2)}</div>
            </div>
            <div>
              <div className="detail-item-label">Return</div>
              <div className="detail-item-value">{returnMultiple}x</div>
            </div>
            <div>
              <div className="detail-item-label">Payout</div>
              <div className="detail-item-value">${payout.toLocaleString()}</div>
            </div>
            <div>
              <div className="detail-item-label">Source</div>
              <div className="detail-item-value">
                {position.contract_source || 'Polymarket'}
                {position.ticker && <span className="detail-ticker">{position.ticker}</span>}
              </div>
            </div>
            {position.end_date_formatted && (
              <div>
                <div className="detail-item-label">Expires</div>
                <div className="detail-item-value">{position.end_date_formatted}</div>
              </div>
            )}
            {position.correlation && (
              <div>
                <div className="detail-item-label">Correlation</div>
                <div className="detail-item-value">
                  <span className={`correlation-badge ${position.correlation.toLowerCase()}`}>
                    {position.correlation}
                  </span>
                </div>
              </div>
            )}
            <div>
              <div className="detail-item-label">Hedges</div>
              <div className="detail-item-value">{position.risk?.event || '—'}</div>
            </div>
          </div>
          <p className="position-reasoning-text">
            {position.reasoning}
          </p>
          {position.url && (
            <a
              href={position.url}
              target="_blank"
              rel="noopener noreferrer"
              className="contract-link"
              onClick={(e) => e.stopPropagation()}
            >
              View on {position.contract_source || 'Polymarket'} <ExternalLink size={11} />
            </a>
          )}
        </div>
      )}
    </motion.div>
  )
}

function RiskCard({ risk }) {
  const severity = risk.confidence >= 0.8 ? 'high' : risk.confidence >= 0.5 ? 'medium' : 'low'

  return (
    <div className="risk-card">
      <div className="risk-header">
        <span className="risk-event">{risk.event}</span>
        <span className={`risk-badge ${severity}`}>{severity}</span>
      </div>
      {risk.exposure_amount && (
        <div className="risk-exposure">
          ${Number(risk.exposure_amount).toLocaleString()} exposure
        </div>
      )}
      <div className="risk-description">{risk.description}</div>
    </div>
  )
}

export default function Results({ data, onReset }) {
  const [showRisks, setShowRisks] = useState(false)
  const positions = data.positions || []
  const totalCost = data.total_cost || 0
  const risks = data.risks || []
  const warnings = data.warnings || []
  const unhedgeable = data.unhedgeable_risks || []

  const hasNotes = warnings.length > 0 || unhedgeable.length > 0
  const hasRisks = risks.length > 0
  const risksLabel = hasRisks
    ? (hasNotes ? `View ${risks.length} risks & notes` : `View ${risks.length} identified risks`)
    : (hasNotes ? `View ${warnings.length + unhedgeable.length} notes` : null)

  return (
    <motion.div
      className="results"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <PortfolioSummary data={data} />

      {positions.length > 0 && (
        <section className="positions-section">
          <div className="section-header">
            Recommended Positions <span className="section-count">{positions.length}</span>
          </div>
          <div className="position-cards">
            {positions.map((pos, i) => (
              <PositionCard key={i} position={pos} index={i} totalCost={totalCost} />
            ))}
          </div>
        </section>
      )}

      {risksLabel && (
        <section className="risks-section">
          <button className="risks-toggle" onClick={() => setShowRisks(!showRisks)}>
            {showRisks ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {risksLabel}
          </button>
          {showRisks && (
            <>
              <div className="risks-grid">
                {risks.map((risk, i) => (
                  <RiskCard key={i} risk={risk} />
                ))}
              </div>
              {hasNotes && (
                <div className="risks-notes">
                  {warnings.map((w, i) => (
                    <div key={`w-${i}`} className="note-item">
                      <AlertTriangle size={13} className="note-icon amber" />
                      <span className="note-text">{w}</span>
                    </div>
                  ))}
                  {unhedgeable.map((u, i) => (
                    <div key={`u-${i}`} className="note-item">
                      <Info size={13} className="note-icon red" />
                      <span className="note-text">
                        {typeof u === 'string' ? u : (
                          <><strong>{u.event}</strong> — {u.reason}</>
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </section>
      )}

      <div className="reset-bar">
        <button className="reset-btn" onClick={onReset}>
          <ArrowLeft size={13} />
          New analysis
        </button>
      </div>
    </motion.div>
  )
}
