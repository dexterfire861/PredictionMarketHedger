import { useMemo } from 'react'

// Generates a deterministic pseudo-random sparkline from a seed string
function generatePoints(seed, count = 20) {
  let hash = 0
  for (let i = 0; i < seed.length; i++) {
    hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0
  }

  const points = []
  let value = 50 + (hash % 30)
  for (let i = 0; i < count; i++) {
    hash = ((hash << 5) - hash + i * 7) | 0
    const delta = (hash % 20) - 10
    value = Math.max(10, Math.min(90, value + delta))
    points.push(value)
  }
  return points
}

export default function Sparkline({ seed = 'default', width = 64, height = 28, color = 'var(--green)' }) {
  const points = useMemo(() => generatePoints(seed), [seed])

  const min = Math.min(...points)
  const max = Math.max(...points)
  const range = max - min || 1

  const pathPoints = points.map((v, i) => {
    const x = (i / (points.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  })

  const d = `M${pathPoints.join(' L')}`

  return (
    <div className="sparkline-wrap">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}
