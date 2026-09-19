import { useEffect, useMemo, useRef, useState } from 'react'
import { datasetOf, generate, getExperiments, modelOf } from '../api/experiments'

/**
 * Chat with a trained character-level GPT.
 *
 * These are tiny character models trained on small corpora, so this is a
 * "prompt -> continuation" chat: the model continues your text in the style
 * of its training data. It will not answer questions like a large LLM.
 */
export default function Chat() {
  const [experiments, setExperiments] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [messages, setMessages] = useState([]) // { role: 'user' | 'model', text }
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  // Generation controls
  const [maxTokens, setMaxTokens] = useState(200)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(20)

  const scrollRef = useRef(null)

  // Only experiments with a saved checkpoint can be chatted with.
  const chattable = useMemo(
    () => experiments.filter((e) => e.has_model),
    [experiments],
  )

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      setLoading(true)
      try {
        const list = await getExperiments()
        if (cancelled) return
        setExperiments(list ?? [])
        const first = (list ?? []).find((e) => e.has_model)
        if (first) setSelectedId(first.id)
        setError(null)
      } catch (e) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    // Auto-scroll to the newest message.
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, busy])

  const selected = experiments.find((e) => e.id === selectedId) ?? null

  const send = async (e) => {
    e?.preventDefault()
    const text = prompt.trim()
    if (!text || selectedId == null || busy) return

    setMessages((prev) => [...prev, { role: 'user', text }])
    setPrompt('')
    setBusy(true)
    setError(null)

    try {
      const res = await generate(selectedId, {
        prompt: text,
        max_new_tokens: Number(maxTokens),
        temperature: Number(temperature),
        top_k: Number(topK),
      })
      setMessages((prev) => [...prev, { role: 'model', text: res.completion ?? '' }])
    } catch (err) {
      setError(err.message)
      setMessages((prev) => [
        ...prev,
        { role: 'model', text: '', failed: true },
      ])
    } finally {
      setBusy(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setError(null)
  }

  return (
    <div className="shell chat-shell">
      <header className="masthead">
        <div>
          <h1>chat with a trained model</h1>
          <p>
            Pick a trained run and give it a prompt. These are small character-level models, so they
            continue your text in the style of their training data rather than answering questions.
          </p>
        </div>
      </header>

      {error && (
        <div className="panel notice">
          <strong>Something went wrong</strong>
          <p>{error}</p>
        </div>
      )}

      <div className="chat-layout">
        <aside className="panel chat-controls">
          <label className="field">
            <span>Model</span>
            {loading ? (
              <div className="empty">Loading…</div>
            ) : chattable.length === 0 ? (
              <div className="empty">
                <strong>No trained models yet</strong>
                Train a run on the Training page, then come back here.
              </div>
            ) : (
              <select
                value={selectedId ?? ''}
                onChange={(e) => { setSelectedId(Number(e.target.value)); clearChat() }}
              >
                {chattable.map((e) => (
                  <option key={e.id} value={e.id}>
                    #{e.id} · {e.name}
                  </option>
                ))}
              </select>
            )}
          </label>

          {selected && (
            <div className="chat-meta num">
              <div><span>Model</span>{modelOf(selected)?.label ?? selected.config?.model_id ?? '—'}</div>
              <div><span>Corpus</span>{datasetOf(selected)?.label ?? selected.config?.dataset_file ?? '—'}</div>
              <div><span>Steps</span>{selected.config?.max_steps ?? '—'}</div>
            </div>
          )}

          <label className="field">
            <span>Max new tokens: {maxTokens}</span>
            <input type="range" min="20" max="1000" step="20"
              value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} />
          </label>

          <label className="field">
            <span>Temperature: {Number(temperature).toFixed(2)}</span>
            <input type="range" min="0.1" max="2" step="0.1"
              value={temperature} onChange={(e) => setTemperature(e.target.value)} />
          </label>

          <label className="field">
            <span>Top-K: {topK}</span>
            <input type="range" min="1" max="50" step="1"
              value={topK} onChange={(e) => setTopK(e.target.value)} />
          </label>

          <button type="button" className="ghost-btn" onClick={clearChat} disabled={!messages.length}>
            Clear conversation
          </button>
        </aside>

        <main className="panel chat-main">
          <div className="chat-log" ref={scrollRef}>
            {messages.length === 0 && !busy && (
              <div className="empty">
                <strong>Start a conversation</strong>
                Type a prompt below and the selected model will continue it.
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`bubble ${m.role}`}>
                <span className="bubble-role">{m.role === 'user' ? 'You' : 'Model'}</span>
                {m.failed ? (
                  <span className="bubble-failed">generation failed — see the error above</span>
                ) : (
                  <span className="bubble-text">{m.text}</span>
                )}
              </div>
            ))}

            {busy && (
              <div className="bubble model">
                <span className="bubble-role">Model</span>
                <span className="bubble-text typing">generating…</span>
              </div>
            )}
          </div>

          <form className="chat-input" onSubmit={send}>
            <textarea
              rows={2}
              placeholder={selectedId == null ? 'Select a trained model first' : 'Enter a prompt…'}
              value={prompt}
              disabled={selectedId == null || busy}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) send(e)
              }}
            />
            <button type="submit" disabled={selectedId == null || busy || !prompt.trim()}>
              {busy ? 'Generating…' : 'Send'}
            </button>
          </form>
        </main>
      </div>
    </div>
  )
}
