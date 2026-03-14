import { useState, useCallback } from 'react'
import { AnimatePresence } from 'framer-motion'
import Header from './components/Header'
import Hero from './components/Hero'
import IntakeForm from './components/IntakeForm'
import LoadingState from './components/LoadingState'
import Results from './components/Results'
import Footer from './components/Footer'
import { MOCK_RESULTS } from './data/mockResults'
import { transformResponse } from './utils/transformResponse'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [businessDescription, setBusinessDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const [results, setResults] = useState(null)

  const handleAnalyze = useCallback(async () => {
    if (!businessDescription.trim()) return

    setLoading(true)
    setLoadingStep(0)
    setResults(null)

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          business_description: businessDescription,
        }),
      })

      if (!response.ok) {
        throw new Error(`Analysis failed (${response.status})`)
      }

      const data = await response.json()
      setResults(transformResponse(data))
    } catch (err) {
      console.error('API call failed, using mock data for demo:', err.message)
      for (let i = 0; i < 5; i++) {
        setLoadingStep(i)
        await new Promise(r => setTimeout(r, 800 + Math.random() * 600))
      }
      setResults(MOCK_RESULTS)
    } finally {
      setLoading(false)
    }
  }, [businessDescription])

  const handleReset = useCallback(() => {
    setResults(null)
    setBusinessDescription('')
  }, [])

  return (
    <div className="app">
      <Header />
      <main className="main">
        {!results && !loading && (
          <>
            <Hero />
            <IntakeForm
              businessDescription={businessDescription}
              onDescriptionChange={setBusinessDescription}
              onAnalyze={handleAnalyze}
              loading={loading}
            />
          </>
        )}

        <AnimatePresence mode="wait">
          {loading && <LoadingState step={loadingStep} key="loading" />}
        </AnimatePresence>

        {results && !loading && (
          <Results data={results} onReset={handleReset} />
        )}
      </main>
      <Footer />
    </div>
  )
}

export default App
