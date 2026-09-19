
import { DATASETS } from '../api/Experiments'

const fmtTokens = (n) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B tok`
  if (n >= 1e6) return `${Math.round(n / 1e6)}M tok`
  return `${Math.round(n / 1e3)}K tok`
}

const maxBytes = Math.max(...DATASETS.map((d) => d.bytes))

export default function DatasetSelector({ value, onChange }) {
  return (
    <div className="options" role="group" aria-label="Training corpus">
      {DATASETS.map((d) => {
        const selected = d.id === value
        // log scale, otherwise 1 MB is an invisible sliver next to 1 GB
        const width = (Math.log10(d.bytes) / Math.log10(maxBytes)) * 100
        return (
          <button
            key={d.id}
            type="button"
            className="option"
            aria-pressed={selected}
            onClick={() => onChange(d.id)}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
                <span className="option-name">{d.label}</span>
                <span className="option-meta" style={{ marginLeft: 0 }}>{fmtTokens(d.tokens)}</span>
              </div>
              <div className="option-note">{d.note}</div>
              <div className="sizebar" style={{ color: 'var(--accent)' }}>
                <i style={{ width: `${width}%` }} />
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}