import { motion } from 'framer-motion'
import { FileText, Search, Shield } from 'lucide-react'
import Ticker from './Ticker'

export default function Hero() {
  return (
    <>
      <motion.section
        className="hero"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
      >
        <h1>
          Hedge your <span className="accent">uninsurable</span> risks
        </h1>
        <p className="hero-sub">
          AI-powered protection using prediction markets.
        </p>
        <div className="how-it-works">
          <div className="how-step">
            <FileText size={16} className="how-icon" />
            <span>Describe your risk</span>
          </div>
          <span className="how-arrow" />
          <div className="how-step">
            <Search size={16} className="how-icon" />
            <span>AI scans markets</span>
          </div>
          <span className="how-arrow" />
          <div className="how-step">
            <Shield size={16} className="how-icon" />
            <span>Get your hedge portfolio</span>
          </div>
        </div>
      </motion.section>
      <Ticker />
    </>
  )
}
