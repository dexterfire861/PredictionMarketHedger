const COLORS = ['#c9a84c', '#3d9970', '#5b8fd4', '#8b6cc1']

export default function DonutChart({ positions = [], size = 64 }) {
  const total = positions.reduce((s, p) => s + (p.allocation || p.cost || 0), 0)
  if (!total) return null

  const r = (size - 8) / 2
  const cx = size / 2
  const cy = size / 2
  const circumference = 2 * Math.PI * r

  let offset = 0
  const segments = positions.map((p, i) => {
    const fraction = (p.allocation || p.cost || 0) / total
    const dashLength = fraction * circumference
    const dashOffset = -offset
    offset += dashLength
    return (
      <circle
        key={i}
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke={COLORS[i % COLORS.length]}
        strokeWidth="6"
        strokeDasharray={`${dashLength} ${circumference - dashLength}`}
        strokeDashoffset={dashOffset}
        strokeLinecap="butt"
        transform={`rotate(-90 ${cx} ${cy})`}
        style={{ transition: 'stroke-dasharray 0.4s ease' }}
      />
    )
  })

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth="6" />
      {segments}
    </svg>
  )
}
