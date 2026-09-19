/**
 * API client for the FastAPI backend (backend/app/api/experiments.py).
 *
 *   GET  /api/experiments
 *   POST /api/experiments                -> { id, ... }
 *   POST /api/experiments/{id}/train     -> { id, status: "queued" }
 *   GET  /api/experiments/{id}
 *   GET  /api/experiments/{id}/metrics   -> MetricPoint[]
 *   WS   /ws/experiments/{id}            -> live MetricPoint stream
 *
 * MetricPoint { step, train_loss, val_loss, tokens_per_sec, tokens_seen, elapsed_s }
 *
 * There is no simulator. If the backend is down, calls reject and the dashboard
 * surfaces the error instead of inventing numbers.
 */

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws')

export const experimentSocketUrl = (id) => `${WS_BASE_URL}/ws/experiments/${id}`

/* ----------------------------------------------------------------- catalogs */

export const DATASETS = [
  { id: '1mb', file: 'data/1mb.txt', label: '1 MB', bytes: 1048576, tokens: 262000, note: 'Smoke test. Overfits in minutes.' },
  { id: '10mb', file: 'data/10mb.txt', label: '10 MB', bytes: 10485760, tokens: 2600000, note: 'Enough signal for small models.' },
  { id: '100mb', file: 'data/100mb.txt', label: '100 MB', bytes: 104857600, tokens: 26000000, note: 'Where scaling starts to show.' },
  { id: '1gb', file: 'data/1gb.txt', label: '1 GB', bytes: 1073741824, tokens: 268000000, note: 'Streams from disk. Long runs.' },
]

export const MODELS = [
  { id: '10k', label: '10K', params: 10000, n_layer: 1, n_head: 1, n_embd: 32, block_size: 64 },
  { id: '100k', label: '100K', params: 100000, n_layer: 2, n_head: 2, n_embd: 64, block_size: 128 },
  { id: '1m', label: '1M', params: 1000000, n_layer: 4, n_head: 4, n_embd: 128, block_size: 256 },
  { id: '10m', label: '10M', params: 10000000, n_layer: 6, n_head: 6, n_embd: 384, block_size: 256 },
  { id: '100m', label: '100M', params: 100000000, n_layer: 12, n_head: 12, n_embd: 768, block_size: 512 },
  { id: '1b', label: '1B', params: 1000000000, n_layer: 24, n_head: 16, n_embd: 2048, block_size: 1024 },
]

// Configs above this are listed but will very likely be rejected by an 8 GB M1.
export const LOCALLY_HEAVY = ['100m', '1b']

export const findDataset = (id) => DATASETS.find((d) => d.id === id)
export const findModel = (id) => MODELS.find((m) => m.id === id)

/** Backend stores model_id in config, so recover the catalog entry from a row. */
export const modelOf = (experiment) => findModel(experiment?.config?.model_id) ?? null
export const datasetOf = (experiment) =>
  DATASETS.find((d) => d.file === experiment?.config?.dataset_file) ?? null

export const isActive = (status) => status === 'queued' || status === 'running' || status === 'created'

/* ------------------------------------------------------------------ helpers */

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(API_BASE_URL + path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error(`Cannot reach the backend at ${API_BASE_URL}. Is uvicorn running?`)
  }

  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body)
    } catch {
      detail = await res.text().catch(() => '')
    }
    throw new Error(detail ? `${res.status} — ${detail}` : `${res.status} ${res.statusText}`)
  }

  return res.status === 204 ? null : res.json()
}

/* ---------------------------------------------------------------- endpoints */

export const getExperiments = () => request('/api/experiments')

export const getExperiment = (id) => request(`/api/experiments/${id}`)

export const getExperimentMetrics = (id) => request(`/api/experiments/${id}/metrics`)

export const createExperiment = (payload) =>
  request('/api/experiments', { method: 'POST', body: JSON.stringify(payload) })

export const startExperiment = (id) =>
  request(`/api/experiments/${id}/train`, { method: 'POST' })

/** Build the POST body from selected config. Kept here so the shape lives in one place. */
export function buildExperimentPayload({ dataset, model, batch_size, learning_rate, max_steps, device }) {
  return {
    name: `${model.label} - ${dataset.label}`,
    description: '',
    config: {
      dataset_file: dataset.file,
      model_id: model.id,
      n_layer: model.n_layer,
      n_head: model.n_head,
      n_embd: model.n_embd,
      block_size: model.block_size,
      batch_size,
      learning_rate,
      max_steps,
      device,
    },
  }
}