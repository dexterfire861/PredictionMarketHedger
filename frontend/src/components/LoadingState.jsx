import { motion } from 'framer-motion'

const STEPS = [
  'Analyzing your risk exposure',
  'Searching Polymarket and Kalshi',
  'Evaluating contract correlations',
  'Building hedge portfolio',
  'Finalizing protection plan',
]

export default function LoadingState({ step }) {
  return (
    <motion.div
      className="loading-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
    >
      <div className="loading-pulse">
        <div className="loading-center-dot" />
      </div>
      <div className="loading-steps">
        {STEPS.map((text, i) => (
          <div
            key={i}
            className={`loading-step ${
              i < step ? 'done' : i === step ? 'active' : ''
            }`}
          >
            <span className="step-indicator">
              <svg className="step-check" width="8" height="8" viewBox="0 0 8 8">
                <path
                  d="M1.5 4L3.5 6L6.5 2"
                  stroke={i < step ? '#343b4f' : '#080b14'}
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
              </svg>
            </span>
            {text}
          </div>
        ))}
      </div>
    </motion.div>
  )
}
