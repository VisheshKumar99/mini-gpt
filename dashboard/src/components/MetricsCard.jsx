const fmt = {
  loss: (v) => (v == null ? '—' : v.toFixed(3)),
  ppl: (v) => (v == null ? '—' : v > 1000 ? v.toExponential(2) : v.toFixed(1)),
  int: (v) => (v == null ? '—' : Math.round(v).toLocaleString()),
  tokens: (v) =>
    v == null ? '—' : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(0)}K`,
  time: (s) => {
    if (s == null) return '—'
    if (s < 60) return `${Math.round(s)}s`
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
    return `${(s / 3600).toFixed(1)}h`
  },
  mem: (v) => (v == null ? '—' : v >= 1024 ? `${(v / 1024).toFixed(1)} GB` : `${Math.round(v)} MB`),
}

/**
 * `run` is an experiment; `points` its metric history.
 * Falls back to the last streamed point while a run is still going.
 */
export default function MetricsCard({ run, points = [] }) {
  if (!run) {
    return (
      <div className="panel">
        <div className="empty">
          <strong>No run selected</strong>
          Pick a corpus and a parameter count, then start a run.
        </div>
      </div>
    )
  }

  const last = points[points.length - 1] ?? {}
  const m = run.metrics ?? {}
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
    { label: 'Wall clock', value: fmt.time(m.wall_clock_s ?? last.elapsed_s) },
    { label: 'Peak memory', value: fmt.mem(m.peak_memory_mb) },
  ]

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>{run.name}</h2>
        <small>
          step <span className="num">{fmt.int(last.step ?? 0)}</span> of{' '}
          <span className="num">{fmt.int(run.hyperparams?.max_steps)}</span>
        </small>
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