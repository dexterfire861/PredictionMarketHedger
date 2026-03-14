/**
 * Transforms backend API response into the shape the frontend expects.
 *
 * Backend returns a flat array of contracts:
 *   [{ title, source, ticker, url, side, current_price, payout_ratio, end_date, correlation, reasoning, confidence }]
 *
 * Frontend expects:
 *   { positions[], risks[], total_cost, total_max_profit, total_exposure, budget_rationale, warnings[], unhedgeable_risks[] }
 */

const DEFAULT_ALLOCATION_PER_POSITION = 2000

function parsePayoutRatio(raw) {
  if (typeof raw === 'number') return raw
  if (typeof raw === 'string') {
    const n = parseFloat(raw.replace('x', ''))
    return isNaN(n) ? 0 : n
  }
  return 0
}

function formatDate(dateStr) {
  if (!dateStr) return null
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  } catch {
    return dateStr
  }
}

function generateWarnings(positions) {
  const warnings = []
  const now = new Date()

  for (const p of positions) {
    if (!p.end_date) continue
    const end = new Date(p.end_date)
    const daysUntil = Math.round((end - now) / (1000 * 60 * 60 * 24))
    if (daysUntil > 0 && daysUntil < 30) {
      warnings.push(
        `${p.contract_title} expires in ${daysUntil} days — short duration may not match ongoing business risk.`
      )
    }
  }

  return warnings
}

function mapContract(contract, allocation) {
  return {
    contract_title: contract.title || contract.contract_title,
    contract_source: contract.source || contract.contract_source || 'Polymarket',
    side: contract.side,
    current_price: contract.current_price,
    allocation,
    url: contract.url || '',
    reasoning: contract.reasoning || '',
    ticker: contract.ticker || null,
    end_date: contract.end_date || null,
    end_date_formatted: formatDate(contract.end_date),
    correlation: contract.correlation || null,
    confidence: contract.confidence || 0,
    payout_ratio_raw: contract.payout_ratio || null,
    risk: {
      event: contract.risk?.event || contract.title || contract.contract_title || 'Identified risk',
    },
  }
}

export function transformResponse(data) {
  // If already in the full frontend format, normalize and pass through
  if (data && !Array.isArray(data) && data.positions) {
    // Enrich existing positions with new fields if present
    const positions = data.positions.map((p) => ({
      ...p,
      contract_title: p.contract_title || p.title,
      contract_source: p.contract_source || p.source || 'Polymarket',
      end_date_formatted: formatDate(p.end_date),
    }))
    return { ...data, positions }
  }

  // Backend returns a flat array of contracts — transform it
  const contracts = Array.isArray(data) ? data : []
  if (contracts.length === 0) {
    return {
      positions: [],
      risks: [],
      total_cost: 0,
      total_max_profit: 0,
      total_exposure: 0,
      budget_rationale: '',
      warnings: [],
      unhedgeable_risks: [],
    }
  }

  // Weight allocation by confidence (higher confidence = more allocation)
  const totalConfidence = contracts.reduce((s, c) => s + (c.confidence || 0.5), 0)
  const baseBudget = contracts.length * DEFAULT_ALLOCATION_PER_POSITION

  const positions = contracts.map((c) => {
    const weight = (c.confidence || 0.5) / totalConfidence
    const allocation = Math.round(baseBudget * weight)
    return mapContract(c, allocation)
  })

  const totalCost = positions.reduce((s, p) => s + p.allocation, 0)
  const totalMaxProfit = positions.reduce((s, p) => {
    const price = p.current_price || 0.5
    return s + (price > 0 ? Math.round(p.allocation / price) : 0)
  }, 0)

  const avgPayout = totalCost > 0 ? (totalMaxProfit / totalCost).toFixed(1) : '0'
  const warnings = generateWarnings(positions)

  return {
    positions,
    risks: [],
    total_cost: totalCost,
    total_max_profit: totalMaxProfit,
    total_exposure: 0,
    budget_rationale: `Recommended allocating $${totalCost.toLocaleString()} across ${positions.length} positions for up to $${totalMaxProfit.toLocaleString()} in max protection (${avgPayout}x average coverage ratio).`,
    warnings,
    unhedgeable_risks: [],
  }
}
