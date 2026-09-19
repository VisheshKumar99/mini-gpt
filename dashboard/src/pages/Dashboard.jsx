import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ExperimentForm from '../components/ExperimentForm'
import LossChart from '../components/LossChart'
import MetricsCard from '../components/MetricsCard'
import { SERIES_COLOR } from '../components/ModelSelector'
import useExperimentWebSocket from '../hooks/useExperimentWebSocket'
import {
  buildExperimentPayload,
  createExperiment,
  datasetOf,
  findDataset,
  findModel,
  getExperiment,
  getExperimentMetrics,
  getExperiments,
  isActive,
  modelOf,
  startExperiment,
} from '../api/experiments'

const fmtTime = (s) => (s == null ? '—' : s < 60 ? `${Number(s).toFixed(1)}s` : `${(s / 60).toFixed(1)}m`)
const fmtTokens = (v) =>
  v == null ? '—' : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(1)}K`

export default function Dashboard() {
  /* ----------------------------------------------- runtime state (not config) */
  const [runs, setRuns] = useState([])
  const [pointsByRun, setPointsByRun] = useState({})
  const [selected, setSelected] = useState([])   // ids plotted on the charts
  const [focused, setFocused] = useState(null)   // id shown in MetricsCard
  const [activeId, setActiveId] = useState(null) // id whose socket is open
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  const loadedMetrics = useRef(new Set())

  /* -------------------------------------------------------------- data loads */

  const loadMetrics = useCallback(async (id) => {
    if (loadedMetrics.current.has(id)) return
    loadedMetrics.current.add(id)
    try {
      const history = await getExperimentMetrics(id)
      setPointsByRun((prev) => ({ ...prev, [id]: history ?? [] }))
    } catch (e) {
      loadedMetrics.current.delete(id)
      setError(`Could not load metrics for experiment ${id}: ${e.message}`)
    }
  }, [])

  const refreshRun = useCallback(async (id) => {
    try {
      const fresh = await getExperiment(id)
      setRuns((prev) => prev.map((r) => (r.id === id ? { ...r, ...fresh } : r)))
      if (!isActive(fresh.status)) {
        setActiveId((current) => (current === id ? null : current))
        // Pick up anything the socket missed between the last frame and the close.
        loadedMetrics.current.delete(id)
        const history = await getExperimentMetrics(id).catch(() => null)
        if (history) setPointsByRun((prev) => ({ ...prev, [id]: history }))
        loadedMetrics.current.add(id)
      }
      if (fresh.status === 'failed') {
        setError(`Experiment ${id} failed on the backend. Check the uvicorn logs for the traceback.`)
      }
      return fresh
    } catch (e) {
      setError(e.message)
      return null
    }
  }, [])

  const loadExperiments = useCallback(async () => {
    setLoading(true)
    try {
      const list = await getExperiments()
      setError(null)
      setRuns(list ?? [])
      if (list?.length) {
        const first = list.slice(0, 3).map((r) => r.id)
        setSelected(first)
        setFocused(list[0].id)
        first.forEach(loadMetrics)
        const running = list.find((r) => isActive(r.status))
        if (running) setActiveId(running.id)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [loadMetrics])

  useEffect(() => { loadExperiments() }, [loadExperiments])

  /* ------------------------------------------------------- live metric stream */

  const appendPoint = useCallback((id, point) => {
    setPointsByRun((prev) => {
      const list = prev[id] ?? []
      if (list.some((p) => p.step === point.step)) return prev
      return { ...prev, [id]: [...list, point].sort((a, b) => a.step - b.step) }
    })
  }, [])

  const seedSteps = useMemo(
    () => (pointsByRun[activeId] ?? []).map((p) => p.step),
    [pointsByRun, activeId],
  )

  const { connection } = useExperimentWebSocket({
    experimentId: activeId,
    enabled: activeId != null,
    seedSteps,
    onMetric: (metric) => {
      appendPoint(activeId, metric)
      setRuns((prev) =>
        prev.map((r) => (r.id === activeId && r.status === 'queued' ? { ...r, status: 'running' } : r)),
      )
    },
    // A message is not a completion signal. Ask the backend for the real status.
    onClose: () => { if (activeId != null) refreshRun(activeId) },
    onError: (msg) => setError(msg),
  })

  /* -------------------------------------------------------------- run actions */

  const startRun = async (config) => {
    setBusy(true)
    setError(null)
    const dataset = findDataset(config.datasetId)
    const model = findModel(config.modelId)

    try {
      const created = await createExperiment(
        buildExperimentPayload({ ...config, dataset, model }),
      )

      setRuns((prev) => [created, ...prev])
      setPointsByRun((prev) => ({ ...prev, [created.id]: [] }))
      loadedMetrics.current.add(created.id)
      setSelected((prev) => [...new Set([created.id, ...prev])].slice(0, 6))
      setFocused(created.id)

      const queued = await startExperiment(created.id)
      setRuns((prev) =>
        prev.map((r) => (r.id === created.id ? { ...r, status: queued?.status ?? 'queued' } : r)),
      )

      setActiveId(created.id) // opens the socket, closing any previous one
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const toggleSelected = (run) => {
    setFocused(run.id)
    loadMetrics(run.id)
    if (isActive(run.status) && activeId !== run.id) setActiveId(run.id)
    setSelected((prev) =>
      prev.includes(run.id) ? prev.filter((x) => x !== run.id) : [...prev, run.id],
    )
  }

  /* ------------------------------------------------------------- derivations */

  const series = useMemo(
    () =>
      runs
        .filter((r) => selected.includes(r.id))
        .map((r) => ({
          id: r.id,
          label: r.name,
          color: SERIES_COLOR[r.config?.model_id] ?? 'var(--accent)',
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
            color: SERIES_COLOR[r.config?.model_id] ?? 'var(--accent)',
            corpus: datasetOf(r)?.label ?? r.config?.dataset_file ?? '—',
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
  const activeCount = runs.filter((r) => isActive(r.status)).length

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
        <div className="panel notice">
          <strong>Something went wrong</strong>
          <p>{error}</p>
          <button className="ghost-btn" onClick={() => { setError(null); loadExperiments() }}>
            Retry
          </button>
        </div>
      )}

      <div className="layout">
        <aside>
          <ExperimentForm onSubmit={startRun} busy={busy} />
        </aside>

        <main className="stack">
          <LossChart series={series} mode="loss" />

          <MetricsCard
            run={focusedRun}
            points={pointsByRun[focused] ?? []}
            live={focused === activeId ? connection : undefined}
          />

          <LossChart series={series} mode="throughput" />

          <div className="panel">
            <div className="panel-head">
              <h2>Comparison</h2>
              <small>{selected.length} of {runs.length} plotted</small>
            </div>

            {loading ? (
              <div className="empty">Loading experiments…</div>
            ) : !runs.length ? (
              <div className="empty">
                <strong>No runs yet</strong>
                Configure one on the left and click Run experiment.
              </div>
            ) : (
              <>
                <div className="roster" style={{ marginBottom: 18 }}>
                  {runs.map((r) => (
                    <button
                      key={r.id}
                      className="run"
                      aria-pressed={selected.includes(r.id)}
                      onClick={() => toggleSelected(r)}
                    >
                      <span
                        className="swatch"
                        style={{ background: SERIES_COLOR[r.config?.model_id] ?? 'var(--accent)' }}
                      />
                      <span>
                        <span className="run-label">{r.name}</span>
                        <br />
                        <span className="run-sub">
                          #{r.id} · {modelOf(r)?.label ?? r.config?.model_id} · lr{' '}
                          {r.config?.learning_rate} · batch {r.config?.batch_size} ·{' '}
                          {r.config?.max_steps} steps · {r.config?.device}
                        </span>
                      </span>
                      {r.id === activeId && connection === 'open' && <span className="live-dot" />}
                      <span className={`status ${r.status}`} style={{ marginLeft: 'auto' }}>
                        {r.status}
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
                          <th>Elapsed</th>
                        </tr>
                      </thead>
                      <tbody>
                        {table.map((t) => (
                          <tr key={t.id}>
                            <td>
                              <span
                                className="swatch"
                                style={{ background: t.color, marginRight: 8 }}
                              />
                              {t.name}
                            </td>
                            <td>{t.corpus}</td>
                            <td style={{
                              color: t.val === bestVal ? 'var(--accent)' : undefined,
                              fontWeight: t.val === bestVal ? 600 : 400,
                            }}>
                              {t.val != null ? Number(t.val).toFixed(3) : '—'}
                            </td>
                            <td>{t.ppl ? t.ppl.toFixed(1) : '—'}</td>
                            <td style={{ color: t.gap > 0.5 ? 'var(--warn)' : undefined }}>
                              {t.gap != null ? t.gap.toFixed(3) : '—'}
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