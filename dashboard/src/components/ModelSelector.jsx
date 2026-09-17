import { MODELS } from '../api/Experiments'

export const SERIES_COLOR = {
  '10k': 'var(--s-10k)',
  '100k': 'var(--s-100k)',
  '1m': 'var(--s-1m)',
  '10m': 'var(--s-10m)',
  '100m': 'var(--s-100m)',
  '1b': 'var(--s-1b)',
}

const maxParams = Math.max(...MODELS.map((m) => m.params))

export default function ModelSelector({ value, onChange }) {
  const picked = MODELS.find((m) => m.id === value)

  return (
    <>
      <div className="options" role="group" aria-label="Model size">
        {MODELS.map((m) => {
          const width = (Math.log10(m.params) / Math.log10(maxParams)) * 100
          return (
            <button
              key={m.id}
              type="button"
              className="option"
              aria-pressed={m.id === value}
              onClick={() => onChange(m.id)}
            >
              <span className="swatch" style={{ background: SERIES_COLOR[m.id] }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
                  <span className="option-name">{m.label}</span>
                  <span className="option-meta" style={{ marginLeft: 0 }}>
                    {m.n_layer}L · {m.n_head}H · {m.n_embd}d
                  </span>
                </div>
                <div className="sizebar" style={{ color: SERIES_COLOR[m.id] }}>
                  <i style={{ width: `${width}%` }} />
                </div>
              </div>
            </button>
          )
        })}
      </div>

      {picked && (
        <div className="config-readout">
          passed to <b>src/gpt.py</b>
          <br />
          n_layer=<b>{picked.n_layer}</b> n_head=<b>{picked.n_head}</b> n_embd=<b>{picked.n_embd}</b>
          <br />
          block_size=<b>{picked.block_size}</b>
        </div>
      )}
    </>
  )
}