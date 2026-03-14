import { motion } from 'framer-motion'

export default function IntakeForm({
  businessDescription,
  onDescriptionChange,
  onAnalyze,
  loading,
}) {
  return (
    <motion.section
      className="intake-section"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1, ease: 'easeOut' }}
    >
      <div className="field-label">Tell us about your business</div>
      <textarea
        className="intake-textarea"
        placeholder="I run a small electronics import business sourcing components from Shenzhen, China. Revenue is $3.2M with 12% margins, and 80% of my supply comes from China..."
        value={businessDescription}
        onChange={(e) => onDescriptionChange(e.target.value)}
      />

      <button
        className="analyze-btn"
        onClick={onAnalyze}
        disabled={!businessDescription.trim() || loading}
      >
        Get Protection
      </button>
    </motion.section>
  )
}
