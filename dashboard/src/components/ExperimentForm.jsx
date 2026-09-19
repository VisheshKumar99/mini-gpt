import { useState } from 'react'
import DatasetSelector from './DatasetSelector'
import ModelSelector from './ModelSelector'
import { findDataset, findModel, LOCALLY_HEAVY } from '../api/Experiments'

const DEFAULTS = {
  batch_size: 8,
  learning_rate: 0.0003,
  max_steps: 100,
  device: 'mps',
}

/**
 * Owns experiment *configuration* state only. Runtime state (id, status,
 * metrics) belongs to Dashboard.
 *
 * onSubmit receives: { datasetId, modelId, batch_size, learning_rate, max_steps, device }
 */
export default function ExperimentForm({ onSubmit, busy }) {
  const [datasetId, setDatasetId] = useState('1mb')
  const [modelId, setModelId] = useState('10k')
  const [hp, setHp] = useState(DEFAULTS)

  const dataset = findDataset(datasetId)
  const model = findModel(modelId)
  const tokensSeen = hp.max_steps * hp.batch_size * model.block_size
  const epochs = tokensSeen / dataset.tokens
  const heavy = LOCALLY_HEAVY.includes(modelId)

  const setNum = (key) => (e) => {
    const v = Number(e.target.value)
    setHp((prev) => ({ ...prev, [key]: Number.isFinite(v) ? v : prev[key] }))
  }

  const start = () =>
    onSubmit({
      datasetId,
      modelId,
      batch_size: hp.batch_size,
      learning_rate: hp.learning_rate,
      max_steps: hp.max_steps,
      device: hp.device,
    })

  return (
    <>
      <div className="panel">
        <div className="panel-head">
          <h2>Corpus</h2>
          <small>data/</small>
        </div>
        <DatasetSelector value={datasetId} onChange={setDatasetId} />
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Parameters</h2>
          <small>src/gpt.py</small>
        </div>
        <ModelSelector value={modelId} onChange={setModelId} />
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Training</h2>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Learning rate</span>
            <input
              type="number" step="0.00001" min="0"
              value={hp.learning_rate} onChange={setNum('learning_rate')}
            />
          </label>
          <label className="field">
            <span>Batch size</span>
            <input type="number" min="1" value={hp.batch_size} onChange={setNum('batch_size')} />
          </label>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Steps</span>
            <input type="number" min="1" value={hp.max_steps} onChange={setNum('max_steps')} />
          </label>
          <label className="field">
            <span>Device</span>
            <select
              value={hp.device}
              onChange={(e) => setHp((prev) => ({ ...prev, device: e.target.value }))}
            >
              <option value="mps">mps</option>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
            </select>
          </label>
        </div>

        <div className="config-readout" style={{ marginBottom: 14 }}>
          sees <b>{(tokensSeen / 1e6).toFixed(2)}M</b> tokens
          {' · '}
          <b>{epochs >= 1 ? `${epochs.toFixed(1)}x` : `${(epochs * 100).toFixed(2)}%`}</b> of corpus
          <br />
          {heavy
            ? 'This config will not fit in 8 GB. The backend will reject it and the error appears here.'
            : epochs > 4
              ? 'Repeats the data enough times to memorise it. Expect validation loss to lift away from training.'
              : epochs < 0.1
                ? 'Barely a first pass. Loss will still be falling when the run ends.'
                : 'Roughly one healthy pass over the corpus.'}
        </div>

        <button className="run-btn" onClick={start} disabled={busy}>
          {busy ? 'Starting...' : 'Run experiment'}
        </button>
      </div>
    </>
  )
}