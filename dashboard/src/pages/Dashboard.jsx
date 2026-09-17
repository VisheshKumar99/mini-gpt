import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import ExperimentForm from '../components/ExperimentForm'
import LossChart from '../components/LossChart'
import MetricsCard from '../components/MetricsCard'
import { SERIES_COLOR } from '../components/ModelSelector'

import {
  createExperiment,
  deleteExperiment,
  getMetrics,
  listExperiments,
  streamMetrics,
} from '../api/experiments'

const fmtTime = (s) => (s == null ? '—' : s < 60 ? `${Math.round(s)}s` : `${(s / 60).toFixed(1)}m`)
const fmtTokens = (v) =>
  v == null ? '—' : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(0)}K`

export default function Dashboard() {
  const [runs, setRuns] = useState([])
  const [pointsByRun, setPointsByRun] = useState({})
  const [selected, setSelected] = useState([])   // ids plotted on the chart
  const [focused, setFocused] = useState(null)   // id shown in MetricsCard
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const subs = useRef({})

  /* ------------------------------------------------------------- bootstrap */

  useEffect(() => {
    let alive = true
    listExperiments()
      .then(async (list) => {
        if (!alive || !list?.length) return
        setRuns(list)
        setSelected(list.slice(0, 3).map((r) => r.id))
        setFocused(list[0].id)
        const histories = await Promise.all(list.map((r) => getMetrics(r.id).catch(() => [])))
        if (!alive) return
        setPointsByRun(Object.fromEntries(list.map((r, i) => [r.id, histories[i]])))
      })
      .catch((e) => setError(e.message))
    return () => {
      alive = false
      Object.values(subs.current).forEach((fn) => fn?.())
    }
  }, [])

  /* --------------------------------------------------------------- actions */

  const appendPoint = useCallback((id, point) => {
    setPointsByRun((prev) => {
      const list = prev[id] ?? []
      if (list.length && list[list.length - 1].step >= point.step) return prev
      return { ...prev, [id]: [...list, point] }
    })
  }, [])

  const startRun = async (config) => {
    setBusy(true)
    setError(null)
    try {
      const exp = await createExperiment(config)
      setRuns((prev) => [exp, ...prev])
      setPointsByRun((prev) => ({ ...prev, [exp.id]: [] }))
      setSelected((prev) => [...new Set([exp.id, ...prev])].slice(0, 6))
      setFocused(exp.id)

      subs.current[exp.id] = streamMetrics(
        exp.id,
        (point) => appendPoint(exp.id, point),
        (finished) => {
          setRuns((prev) => prev.map((r) => (r.id === exp.id ? { ...r, ...finished } : r)))
          subs.current[exp.id]?.()
          delete subs.current[exp.id]
        },
      )
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const removeRun = async (id, e) => {
    e.stopPropagation()
    subs.current[id]?.()
    delete subs.current[id]
    setRuns((prev) => prev.filter((r) => r.id !== id))
    setSelected((prev) => prev.filter((x) => x !== id))
    setFocused((prev) => (prev === id ? null : prev))
    await deleteExperiment(id).catch(() => {})
  }

  const toggleSelected = (id) => {
    setFocused(id)
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }

  /* ----------------------------------------------------------- derivations */

  const series = useMemo(
    () =>
      runs
        .filter((r) => selected.includes(r.id))
        .map((r) => ({
          id: r.id,
          label: r.name,
          color: SERIES_COLOR[r.model?.id] ?? 'var(--accent)',
          points: pointsByRun[r.id] ?? [],
        })),
    [runs, selected, pointsByRun],
  )

  const focusedRun = runs.find((r) => r.id === focused) ?? null

  const table = useMemo(
    () =>
      runs
        .filter((r) => selected.includes(r.id))
        .map((r) => {
          const pts = pointsByRun[r.id] ?? []
          const last = pts[pts.length - 1] ?? {}
          const val = r.metrics?.final_val_loss ?? last.val_loss
          const train = r.metrics?.final_train_loss ?? last.train_loss
          return {
            id: r.id,
            name: r.name,
            color: SERIES_COLOR[r.model?.id] ?? 'var(--accent)',
            params: r.model?.params,
            corpus: r.dataset?.label,
            val,
            gap: val != null && train != null ? val - train : null,
            ppl: val != null ? Math.exp(val) : null,
            tps: r.metrics?.tokens_per_sec ?? last.tokens_per_sec,
            time: r.metrics?.wall_clock_s ?? last.elapsed_s,
            tokens: r.metrics?.tokens_seen ?? last.tokens_seen,
          }
        }),
    [runs, selected, pointsByRun],
  )

  const bestVal = Math.min(...table.map((t) => t.val ?? Infinity))
  const activeCount = runs.filter((r) => r.status === 'running').length

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>micro-gpt training runs</h1>
          <p>
            Train the same architecture at six parameter counts across four corpus sizes, and watch
            where extra data stops paying for extra parameters.
          </p>
        </div>
        <div className="masthead-stat">
          <strong className="num">{runs.length}</strong>
          runs logged{activeCount ? ` · ${activeCount} running` : ''}
        </div>
      </header>

      {error && (
        <div className="panel" style={{ borderColor: 'var(--warn)', marginBottom: 20 }}>
          <strong>Run could not start.</strong> {error} Check that the FastAPI server is up on
          port 8000.
        </div>
      )}

      <div className="layout">
        <aside>
          <ExperimentForm onSubmit={startRun} busy={busy} />
        </aside>

        <main className="stack">
          <LossChart series={series} />

          <MetricsCard run={focusedRun} points={pointsByRun[focused] ?? []} />

          <div className="panel">
            <div className="panel-head">
              <h2>Comparison</h2>
              <small>{selected.length} of {runs.length} plotted</small>
            </div>

            {!runs.length ? (
              <div className="empty">
                <strong>No runs yet</strong>
                Start one on the left and it will appear here.
              </div>
            ) : (
              <>
                <div className="roster" style={{ marginBottom: 18 }}>
                  {runs.map((r) => (
                    <button
                      key={r.id}
                      className="run"
                      aria-pressed={selected.includes(r.id)}
                      onClick={() => toggleSelected(r.id)}
                    >
                      <span
                        className="swatch"
                        style={{ background: SERIES_COLOR[r.model?.id] ?? 'var(--accent)' }}
                      />
                      <span>
                        <span className="run-label">{r.name}</span>
                        <br />
                        <span className="run-sub">
                          lr {r.hyperparams?.learning_rate} · batch {r.hyperparams?.batch_size} ·{' '}
                          {r.hyperparams?.max_steps} steps
                        </span>
                      </span>
                      <span className={`status ${r.status}`}>{r.status}</span>
                      <span
                        role="button"
                        tabIndex={0}
                        className="run-kill"
                        title="Remove run"
                        onClick={(e) => removeRun(r.id, e)}
                        onKeyDown={(e) => e.key === 'Enter' && removeRun(r.id, e)}
                      >
                        ×
                      </span>
                    </button>
                  ))}
                </div>

                {table.length > 0 && (
                  <div style={{ overflowX: 'auto' }}>
                    <table className="cmp">
                      <thead>
                        <tr>
                          <th>Run</th>
                          <th>Corpus</th>
                          <th>Val loss</th>
                          <th>Perplexity</th>
                          <th>Gap</th>
                          <th>Tokens</th>
                          <th>Tok/s</th>
                          <th>Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {table.map((t) => (
                          <tr key={t.id}>
                            <td>
                              <span
                                className="swatch"
                                style={{ background: t.color, display: 'inline-block', marginRight: 8 }}
                              />
                              {t.name}
                            </td>
                            <td>{t.corpus}</td>
                            <td style={{ color: t.val === bestVal ? 'var(--accent)' : undefined, fontWeight: t.val === bestVal ? 600 : 400 }}>
                              {t.val?.toFixed(3) ?? '—'}
                            </td>
                            <td>{t.ppl ? t.ppl.toFixed(1) : '—'}</td>
                            <td style={{ color: t.gap > 0.5 ? 'var(--warn)' : undefined }}>
                              {t.gap?.toFixed(3) ?? '—'}
                            </td>
                            <td>{fmtTokens(t.tokens)}</td>
                            <td>{t.tps ? Math.round(t.tps).toLocaleString() : '—'}</td>
                            <td>{fmtTime(t.time)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}