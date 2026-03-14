const ITEMS = [
  'Tariff Escalation',
  'Supply Chain Disruption',
  'Currency Volatility',
  'Geopolitical Instability',
  'Regulatory Change',
  'Commodity Price Shock',
  'Trade War',
  'Sanctions Risk',
  'Interest Rate Shift',
  'Climate Event',
  'Pandemic Disruption',
  'Energy Crisis',
]

export default function Ticker() {
  const doubled = [...ITEMS, ...ITEMS]

  return (
    <div className="ticker-wrap">
      <div className="ticker-track">
        {doubled.map((item, i) => (
          <span className="ticker-item" key={i}>
            <span className="ticker-dot" />
            <span className="name">{item}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
