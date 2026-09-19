import { useState } from 'react'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const AXES = [
  { id: 'step', label: 'Steps', key: 'step' },
  { id: 'tokens', label: 'Tokens seen', key: 'tokens_seen' },
]

const shortNum = (v) => {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `${(v / 1e6).toFixed(0)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`
  return `${v}`
}

function ChartTooltip({ active, payload, xKey, decimals }) {
  if (!active || !payload?.length) return null
  const x = payload[0].payload?.[xKey]
  return (
    <div className="tooltip">
      <div className="t-step">{shortNum(x ?? 0)} {xKey === 'step' ? 'steps' : 'tokens'}</div>
      {payload.map((p) => (
        <div className="t-row" key={p.name} style={{ color: p.stroke }}>
          <span>{p.name}</span>
          <span>{decimals === 0 ? Math.round(p.value).toLocaleString() : Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * series: [{ id, label, color, points: [{ step, tokens_seen, train_loss, val_loss, tokens_per_sec }] }]
 * mode:   'loss' (train + val) or 'throughput' (tokens_per_sec)
 *
 * Points come straight from the backend: GET /api/experiments/{id}/metrics for
 * history, then the WebSocket stream appends to the same array.
 */
export default function LossChart({ series = [], mode = 'loss', title }) {
  const [xAxis, setXAxis] = useState('step')
  const [showTrain, setShowTrain] = useState(true)
  const [showVal, setShowVal] = useState(true)
  const [logY, setLogY] = useState(false)

  const isLoss = mode === 'loss'
  const xKey = AXES.find((a) => a.id === xAxis).key
  const hasData = series.some((s) => s.points?.length)

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>{title ?? (isLoss ? 'Loss' : 'Throughput')}</h2>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {AXES.map((a) => (
            <button
              key={a.id}
              className="ghost-btn"
              aria-pressed={xAxis === a.id}
              onClick={() => setXAxis(a.id)}
            >
              {a.label}
            </button>
          ))}
          {isLoss && (
            <>
              <button className="ghost-btn" aria-pressed={showTrain} onClick={() => setShowTrain((v) => !v)}>Train</button>
              <button className="ghost-btn" aria-pressed={showVal} onClick={() => setShowVal((v) => !v)}>Validation</button>
              <button className="ghost-btn" aria-pressed={logY} onClick={() => setLogY((v) => !v)}>Log y</button>
            </>
          )}
        </div>
      </div>

      {!hasData ? (
        <div className="empty">
          <strong>Nothing plotted yet</strong>
          {isLoss
            ? 'Loss appears here as soon as the first metric arrives over the socket.'
            : 'Tokens per second appears here once training starts.'}
        </div>
      ) : (
        <>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
                <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" vertical={false} />
                <XAxis
                  dataKey={xKey}
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={shortNum}
                  stroke="var(--muted)"
                  tick={{ fontSize: 11, fontFamily: 'var(--mono)' }}
                  label={{
                    value: AXES.find((a) => a.id === xAxis).label,
                    position: 'insideBottom',
                    offset: -12,
                    fill: 'var(--muted)',
                    fontSize: 11,
                  }}
                />
                <YAxis
                  scale={isLoss && logY ? 'log' : 'linear'}
                  domain={isLoss && logY ? ['auto', 'auto'] : ['dataMin - 0.15', 'dataMax + 0.15']}
                  tickFormatter={(v) => (isLoss ? v.toFixed(2) : shortNum(v))}
                  stroke="var(--muted)"
                  tick={{ fontSize: 11, fontFamily: 'var(--mono)' }}
                  width={52}
                />
                <Tooltip content={<ChartTooltip xKey={xKey} decimals={isLoss ? 3 : 0} />} />

                {series.flatMap((s) => {
                  if (!isLoss) {
                    return [
                      <Line
                        key={`${s.id}-tps`}
                        data={s.points}
                        dataKey="tokens_per_sec"
                        name={`${s.label} · tok/s`}
                        stroke={s.color}
                        strokeWidth={1.75}
                        dot={false}
                        isAnimationActive={false}
                      />,
                    ]
                  }
                  const lines = []
                  if (showTrain) {
                    lines.push(
                      <Line
                        key={`${s.id}-train`}
                        data={s.points}
                        dataKey="train_loss"
                        name={`${s.label} · train`}
                        stroke={s.color}
                        strokeWidth={1.75}
                        dot={false}
                        isAnimationActive={false}
                      />,
                    )
                  }
                  if (showVal) {
                    lines.push(
                      <Line
                        key={`${s.id}-val`}
                        data={s.points}
                        dataKey="val_loss"
                        name={`${s.label} · val`}
                        stroke={s.color}
                        strokeWidth={1.75}
                        strokeDasharray="4 3"
                        dot={false}
                        isAnimationActive={false}
                      />,
                    )
                  }
                  return lines
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="legend">
            {series.map((s) => (
              <span key={s.id}>
                <i className="swatch" style={{ background: s.color }} />
                {s.label}
              </span>
            ))}
            {isLoss && <span style={{ marginLeft: 'auto' }}>solid = train, dashed = validation</span>}
          </div>
        </>
      )}
    </div>
  )
}