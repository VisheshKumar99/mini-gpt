import { useState } from 'react'
import DatasetSelector from './DataSelector'
import ModelSelector from './ModelSelector'
import { findDataset, findModel } from '../api/Experiments'

const DEFAULTS = {
  learning_rate: 3e-4,
  batch_size: 32,
  max_steps: 2000,
  eval_interval: 100,
}

export default function ExperimentForm({ onSubmit, busy }) {
  const [datasetId, setDatasetId] = useState('10mb')
  const [modelId, setModelId] = useState('1m')
  const [hp, setHp] = useState(DEFAULTS)

  const dataset = findDataset(datasetId)
  const model = findModel(modelId)
  const tokensSeen = hp.max_steps * hp.batch_size * model.block_size
  const epochs = tokensSeen / dataset.tokens

  const set = (key) => (e) => {
    const v = Number(e.target.value)
    setHp((prev) => ({ ...prev, [key]: Number.isFinite(v) ? v : prev[key] }))
  }

  const start = () => {
    onSubmit({
      name: `${model.label} · ${dataset.label}`,
      dataset_id: datasetId,
      model_id: modelId,
      dataset_path: dataset.file,
      model_config: {
        n_layer: model.n_layer,
        n_head: model.n_head,
        n_embd: model.n_embd,
        block_size: model.block_size,
      },
      hyperparams: hp,
    })
  }

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
              value={hp.learning_rate} onChange={set('learning_rate')}
            />
          </label>
          <label className="field">
            <span>Batch size</span>
            <input type="number" min="1" value={hp.batch_size} onChange={set('batch_size')} />
          </label>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Steps</span>
            <input type="number" min="1" value={hp.max_steps} onChange={set('max_steps')} />
          </label>
          <label className="field">
            <span>Eval every</span>
            <input type="number" min="1" value={hp.eval_interval} onChange={set('eval_interval')} />
          </label>
        </div>

        <div className="config-readout" style={{ marginBottom: 14 }}>
          sees <b>{(tokensSeen / 1e6).toFixed(1)}M</b> tokens
          {' · '}
          <b>{epochs >= 1 ? `${epochs.toFixed(1)}×` : `${(epochs * 100).toFixed(0)}%`}</b> of corpus
          <br />
          {epochs > 4
            ? 'Repeats the data enough times to memorise it — expect the validation curve to lift away from training.'
            : epochs < 0.1
              ? 'Barely a first pass. Loss will still be falling when the run ends.'
              : 'Roughly one healthy pass over the corpus.'}
        </div>

        <button className="run-btn" onClick={start} disabled={busy}>
          {busy ? 'Starting…' : 'Start run'}
        </button>
      </div>
    </>
  )
}