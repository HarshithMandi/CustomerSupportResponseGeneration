import React, { useMemo, useState } from 'react'
import { generateResponse } from './api.js'

const MODE_DEFAULTS = {
  strict: { temperature: 0.2, max_tokens: 150 },
  friendly: { temperature: 0.7, max_tokens: 200 }
}

function clamp(n, min, max) {
  if (Number.isNaN(n)) return min
  return Math.min(max, Math.max(min, n))
}

function sanitizeAnswer(text) {
  if (typeof text !== 'string') return ''

  let t = text
  // Remove <think> blocks if present
  t = t.replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '')
  t = t.replace(/<\/?think>/gi, '')

  // If the model includes markers like "Final answer:" / "Answer:" / "Response:", keep only what follows.
  const re = /^\s*(final\s*answer|answer|response)\s*:\s*/gim
  let m
  let lastEnd = -1
  while ((m = re.exec(t)) !== null) {
    lastEnd = re.lastIndex
  }
  if (lastEnd >= 0) t = t.slice(lastEnd)

  return t.trim()
}

export default function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('strict')

  const defaults = useMemo(() => MODE_DEFAULTS[mode] || MODE_DEFAULTS.strict, [mode])
  const [temperature, setTemperature] = useState(defaults.temperature)
  const [maxTokens, setMaxTokens] = useState(defaults.max_tokens)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  function handleModeChange(nextMode) {
    setMode(nextMode)
    const d = MODE_DEFAULTS[nextMode] || MODE_DEFAULTS.strict
    setTemperature(d.temperature)
    setMaxTokens(d.max_tokens)
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setResult(null)

    const q = query.trim()
    if (!q) {
      setError('Please enter a customer complaint.')
      return
    }

    const tempNum = clamp(Number(temperature), 0, 1)
    const maxNum = clamp(parseInt(maxTokens, 10), 1, 1000)

    setLoading(true)
    try {
      const data = await generateResponse({
        query: q,
        mode,
        temperature: tempNum,
        max_tokens: maxNum
      })
      setResult(data)
    } catch (err) {
      setError(err?.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="chrome" />
      <main className="shell">
        <header className="header">
          <div>
            <h1 className="title">AI Support Response Generator</h1>
            <p className="subtitle">BM25 policy retrieval + Sarvam AI</p>
          </div>
        </header>

        <section className="panel">
          <form className="form" onSubmit={onSubmit}>
            <label className="label">
              Customer complaint
              <textarea
                className="textarea"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., My product arrived late and damaged. Can I get a refund?"
                rows={5}
              />
            </label>

            <div className="row">
              <label className="label">
                Mode
                <select className="select" value={mode} onChange={(e) => handleModeChange(e.target.value)}>
                  <option value="strict">Strict</option>
                  <option value="friendly">Friendly</option>
                </select>
              </label>

              <label className="label">
                Temperature
                <input
                  className="input"
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                />
              </label>

              <label className="label">
                Max tokens
                <input
                  className="input"
                  type="number"
                  min="1"
                  max="1000"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                />
              </label>
            </div>

            <div className="actions">
              <button className="button" type="submit" disabled={loading}>
                {loading ? 'Generating…' : 'Generate response'}
              </button>
              <button
                className="button ghost"
                type="button"
                disabled={loading}
                onClick={() => {
                  setQuery('')
                  setError('')
                  setResult(null)
                  handleModeChange('strict')
                }}
              >
                Reset
              </button>
            </div>

            {error ? <div className="error">{error}</div> : null}
          </form>
        </section>

        <section className="grid">
          <div className="panel">
            <h2 className="h2">AI response</h2>
            <div className="output">
              {result ? (
                <pre className="pre">{sanitizeAnswer(result.response)}</pre>
              ) : (
                <div className="muted">Your generated reply will appear here.</div>
              )}
            </div>
            {result ? (
              <div className="meta">
                <span className="tag">mode: {result.used_mode}</span>
                <span className="tag">temp: {result.used_temperature}</span>
                <span className="tag">max_tokens: {result.used_max_tokens}</span>
                {result.fallback ? <span className="tag warn">fallback</span> : null}
              </div>
            ) : null}
          </div>

          <div className="panel">
            <h2 className="h2">Retrieved policy documents (BM25)</h2>
            <div className="docs">
              {result?.retrieved_docs?.length ? (
                result.retrieved_docs.map((d, idx) => (
                  <details className="doc" key={`${d.title}-${idx}`} open={idx === 0}>
                    <summary className="docSummary">
                      <span className="docTitle">{d.title}</span>
                      <span className="docScore">score: {Number(d.score).toFixed(3)}</span>
                    </summary>
                    <div className="docBody">{d.content}</div>
                  </details>
                ))
              ) : (
                <div className="muted">Top 3 matching policies will show up here.</div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
