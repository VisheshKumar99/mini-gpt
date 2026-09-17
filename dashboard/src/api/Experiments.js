/**
 * Talks to backend/app/api/experiments.py.
 *
 * Expected contract:
 *   GET    /api/experiments                 -> Experiment[]
 *   POST   /api/experiments                 -> Experiment        (starts a run)
 *   GET    /api/experiments/:id             -> Experiment
 *   GET    /api/experiments/:id/metrics     -> MetricPoint[]
 *   DELETE /api/experiments/:id             -> { ok: true }
 *   WS     /ws/experiments/:id              -> MetricPoint stream
 *
 * Experiment  { id, name, status, dataset, model, hyperparams, metrics, created_at }
 * MetricPoint { step, train_loss, val_loss, tokens_per_sec, elapsed_s }
 *
 * If the backend is unreachable every call falls back to a local simulator so the
 * UI is usable standalone. Flip USE_MOCK to true to force it.
 */

const BASE = '/api'
const USE_MOCK = false

/* ----------------------------------------------------------------- catalogs */

export const DATASETS = [
  { id: '1mb', file: 'data/1mb.txt', label: '1 MB', bytes: 1_048_576, tokens: 262_000, note: 'Smoke test. Overfits in minutes.' },
  { id: '10mb', file: 'data/10mb.txt', label: '10 MB', bytes: 10_485_760, tokens: 2_600_000, note: 'Enough signal for small models.' },
  { id: '100mb', file: 'data/100mb.txt', label: '100 MB', bytes: 104_857_600, tokens: 26_000_000, note: 'Where scaling starts to show.' },
  { id: '1gb', file: 'data/1gb.txt', label: '1 GB', bytes: 1_073_741_824, tokens: 268_000_000, note: 'Streams from disk. Long runs.' },
]

// n_layer / n_head / n_embd chosen so the param count lands near the label.
export const MODELS = [
  { id: '10k', label: '10K', params: 10_000, n_layer: 1, n_head: 1, n_embd: 32, block_size: 64 },
  { id: '100k', label: '100K', params: 100_000, n_layer: 2, n_head: 2, n_embd: 64, block_size: 128 },
  { id: '1m', label: '1M', params: 1_000_000, n_layer: 4, n_head: 4, n_embd: 128, block_size: 256 },
  { id: '10m', label: '10M', params: 10_000_000, n_layer: 6, n_head: 6, n_embd: 384, block_size: 256 },
  { id: '100m', label: '100M', params: 100_000_000, n_layer: 12, n_head: 12, n_embd: 768, block_size: 512 },
  { id: '1b', label: '1B', params: 1_000_000_000, n_layer: 24, n_head: 16, n_embd: 2048, block_size: 1024 },
]

export const findDataset = (id) => DATASETS.find((d) => d.id === id)
export const findModel = (id) => MODELS.find((m) => m.id === id)

/* ------------------------------------------------------------------ helpers */

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`)
  }
  if (res.status === 204) return null
  return res.json()
}

async function withFallback(fn, mockFn) {
  if (USE_MOCK) return mockFn()
  try {
    return await fn()
  } catch (err) {
    console.warn('[experiments] backend unavailable, using simulated data:', err.message)
    return mockFn()
  }
}

/* ---------------------------------------------------------------- endpoints */

export function listExperiments() {
  return withFallback(() => request('/experiments'), () => mock.list())
}

export function getExperiment(id) {
  return withFallback(() => request(`/experiments/${id}`), () => mock.get(id))
}

export function getMetrics(id) {
  return withFallback(() => request(`/experiments/${id}/metrics`), () => mock.metrics(id))
}

export function createExperiment(config) {
  return withFallback(
    () => request('/experiments', { method: 'POST', body: JSON.stringify(config) }),
    () => mock.create(config),
  )
}

export function deleteExperiment(id) {
  return withFallback(
    () => request(`/experiments/${id}`, { method: 'DELETE' }),
    () => mock.remove(id),
  )
}

/**
 * Subscribe to live metrics for a running experiment.
 * Returns an unsubscribe function. Falls back to polling if the socket fails.
 */
export function streamMetrics(id, onPoint, onDone) {
  // Runs created while the backend was down live only in the simulator.
  if (USE_MOCK || String(id).startsWith('sim-')) return mock.stream(id, onPoint, onDone)

  let closed = false
  let poll = null
  let socket = null

  const startPolling = () => {
    if (closed || poll) return
    poll = setInterval(async () => {
      const exp = await getExperiment(id).catch(() => null)
      if (!exp) return
      const points = await getMetrics(id).catch(() => [])
      points.forEach(onPoint)
      if (exp.status !== 'running' && exp.status !== 'queued') {
        clearInterval(poll)
        poll = null
        onDone?.(exp)
      }
    }, 2000)
  }

  try {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    socket = new WebSocket(`${proto}://${window.location.host}/ws/experiments/${id}`)
    socket.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'done') onDone?.(msg.experiment)
      else onPoint(msg)
    }
    socket.onerror = startPolling
  } catch {
    startPolling()
  }

  return () => {
    closed = true
    if (poll) clearInterval(poll)
    socket?.close()
  }
}

/* ------------------------------------------------------- simulated backend  */
/* Chinchilla-ish curve: loss falls with params and with tokens seen, and
   small models on huge data plateau while big models on tiny data overfit.   */

const store = new Map()
let counter = 0

function predictedLoss(params, tokens, step, totalSteps) {
  const a = 406.4 / Math.pow(params, 0.34)
  const b = 410.7 / Math.pow(tokens, 0.28)
  const floor = 1.69 + a + b
  const progress = step / totalSteps
  const start = 4.2 + Math.log10(params) * 0.05
  return floor + (start - floor) * Math.exp(-3.4 * progress)
}

const mock = {
  list: () => Promise.resolve([...store.values()].sort((x, y) => y.created_at - x.created_at)),
  get: (id) => Promise.resolve(store.get(id) ?? null),
  metrics: (id) => Promise.resolve(store.get(id)?.points ?? []),
  remove: (id) => {
    store.delete(id)
    return Promise.resolve({ ok: true })
  },
  create: (config) => {
    const model = findModel(config.model_id)
    const dataset = findDataset(config.dataset_id)
    const id = `sim-${++counter}`
    const exp = {
      id,
      name: config.name || `${model.label} · ${dataset.label}`,
      status: 'running',
      dataset: { ...dataset },
      model: { ...model },
      hyperparams: config.hyperparams,
      metrics: {},
      points: [],
      created_at: Date.now(),
      simulated: true,
    }
    store.set(id, exp)
    return Promise.resolve(exp)
  },
  stream: (id, onPoint, onDone) => {
    const exp = store.get(id)
    if (!exp) return () => {}
    const totalSteps = exp.hyperparams.max_steps
    const every = Math.max(1, Math.round(totalSteps / 60))
    // rough throughput model: bigger model = fewer tokens/sec
    const tps = Math.round(3.6e11 / exp.model.params)
    let step = 0
    const t0 = Date.now()

    const tick = setInterval(() => {
      step += every
      if (step > totalSteps) {
        clearInterval(tick)
        exp.status = 'completed'
        const last = exp.points[exp.points.length - 1]
        exp.metrics = {
          final_train_loss: last.train_loss,
          final_val_loss: last.val_loss,
          perplexity: Math.exp(last.val_loss),
          tokens_per_sec: tps,
          tokens_seen: last.tokens_seen,
          wall_clock_s: last.elapsed_s,
          peak_memory_mb: Math.round(exp.model.params * 16 / 1e6) + 120,
        }
        onDone?.(exp)
        return
      }
      const seen = step * exp.hyperparams.batch_size * exp.model.block_size
      const train = predictedLoss(exp.model.params, exp.dataset.tokens, step, totalSteps)
      // overfitting gap grows when the model can memorise the dataset
      const capacity = (exp.model.params * 20) / exp.dataset.tokens
      const gap = Math.min(1.6, capacity * 0.25) * (step / totalSteps)
      const point = {
        step,
        train_loss: +(train + (Math.random() - 0.5) * 0.05).toFixed(4),
        val_loss: +(train + gap + (Math.random() - 0.5) * 0.05).toFixed(4),
        tokens_seen: seen,
        tokens_per_sec: Math.round(tps * (0.95 + Math.random() * 0.1)),
        elapsed_s: Math.round((Date.now() - t0) / 1000),
      }
      exp.points.push(point)
      onPoint(point)
    }, 120)

    return () => clearInterval(tick)
  },
}