const fmt = {
  loss: (v) => (v == null ? '—' : Number(v).toFixed(3)),
  ppl: (v) => (v == null ? '—' : v > 1000 ? v.toExponential(2) : v.toFixed(1)),
  int: (v) => (v == null ? '—' : Math.round(v).toLocaleString()),
  tokens: (v) =>
    v == null ? '—' : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(1)}K`,
  time: (s) => {
    if (s == null) return '—'
    if (s < 60) return `${Number(s).toFixed(1)}s`
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
    return `${(s / 3600).toFixed(1)}h`
  },
}

/**
 * run    — experiment row from the backend ({ id, name, status, config, metrics })
 * points — MetricPoint[] from GET /metrics plus anything the socket appended
 * live   — WebSocket connection state, for the indicator next to the status
 */
export default function MetricsCard({ run, points = [], live }) {
  if (!run) {
    return (
      <div className="panel">
        <div className="empty">
          <strong>No run selected</strong>
          Pick a corpus and a parameter count, then run an experiment.
        </div>
      </div>
    )
  }

  const last = points[points.length - 1] ?? {}
  const m = run.metrics ?? {}
  const maxSteps = run.config?.max_steps ?? 0
  const step = last.step ?? 0
  const progress = maxSteps ? Math.min(100, (step / maxSteps) * 100) : 0

  // Backend fills `metrics` on completion; before that, read the newest point.
  const valLoss = m.final_val_loss ?? last.val_loss
  const trainLoss = m.final_train_loss ?? last.train_loss
  const ppl = m.perplexity ?? (valLoss != null ? Math.exp(valLoss) : null)
  const gap = valLoss != null && trainLoss != null ? valLoss - trainLoss : null

  const tiles = [
    { label: 'Validation loss', value: fmt.loss(valLoss), best: true },
    { label: 'Training loss', value: fmt.loss(trainLoss) },
    { label: 'Perplexity', value: fmt.ppl(ppl) },
    {
      label: 'Overfit gap',
      value: fmt.loss(gap),
      note: gap == null ? '' : gap > 0.5 ? 'memorising' : gap > 0.15 ? 'drifting' : 'healthy',
    },
    { label: 'Throughput', value: fmt.int(m.tokens_per_sec ?? last.tokens_per_sec), note: 'tokens/sec' },
    { label: 'Tokens seen', value: fmt.tokens(m.tokens_seen ?? last.tokens_seen) },
    { label: 'Elapsed', value: fmt.time(m.wall_clock_s ?? last.elapsed_s) },
    { label: 'Steps', value: `${fmt.int(step)} / ${fmt.int(maxSteps)}` },
  ]

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>{run.name}</h2>
        <small style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {live === 'open' && <span className="live-dot" title="Live metrics connected" />}
          <span className={`status ${run.status}`}>{run.status}</span>
        </small>
      </div>

      <div className="progress" aria-label="Training progress">
        <div
          className="progress-bar"
          role="progressbar"
          aria-valuenow={Math.round(progress)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <i style={{ width: `${progress}%` }} />
        </div>
        <span className="num">{progress.toFixed(0)}%</span>
      </div>

      <div className="metrics-grid">
        {tiles.map((t) => (
          <div key={t.label} className={`metric${t.best ? ' is-best' : ''}`}>
            <span>{t.label}</span>
            <strong>{t.value}</strong>
            {t.note && <em>{t.note}</em>}
          </div>
        ))}
      </div>
    </div>
  )
}