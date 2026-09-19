import { useEffect, useRef, useState } from 'react'
import { experimentSocketUrl } from '../api/Experiments'

/**
 * Subscribes to ws://<backend>/ws/experiments/{id} and hands each metric to
 * onMetric. One socket at a time: changing experimentId closes the previous one,
 * and unmounting closes the current one.
 *
 * Duplicate steps are dropped. seedSteps lets the caller pass the steps already
 * loaded from GET /metrics so a reconnect doesn't re-append history.
 *
 * @param {object}   opts
 * @param {number|string|null} opts.experimentId  null disconnects
 * @param {boolean}  opts.enabled                 gate on status being active
 * @param {number[]} opts.seedSteps               steps already in the chart
 * @param {Function} opts.onMetric                (metric) => void
 * @param {Function} opts.onClose                 (event) => void — good place to GET the final status
 * @param {Function} opts.onError                 (message) => void
 * @returns {{ connection: 'idle'|'connecting'|'open'|'closed'|'error', error: string|null }}
 */
export default function useExperimentWebSocket({
  experimentId,
  enabled = true,
  seedSteps,
  onMetric,
  onClose,
  onError,
}) {
  const [connection, setConnection] = useState('idle')
  const [error, setError] = useState(null)

  // Callbacks live in refs so a re-render doesn't tear down the socket.
  const handlers = useRef({ onMetric, onClose, onError })
  handlers.current = { onMetric, onClose, onError }

  const seedRef = useRef(seedSteps)
  seedRef.current = seedSteps

  useEffect(() => {
    if (!experimentId || !enabled) {
      setConnection('idle')
      return undefined
    }

    let socket
    let disposed = false
    const seen = new Set(seedRef.current ?? [])

    setError(null)
    setConnection('connecting')

    try {
      socket = new WebSocket(experimentSocketUrl(experimentId))
    } catch {
      const msg = `Could not open a WebSocket for experiment ${experimentId}.`
      setConnection('error')
      setError(msg)
      handlers.current.onError?.(msg)
      return undefined
    }

    socket.onopen = () => {
      if (!disposed) setConnection('open')
    }

    socket.onmessage = (event) => {
      if (disposed) return
      let metric
      try {
        metric = JSON.parse(event.data)
      } catch {
        return // ignore anything that isn't a JSON metric frame
      }
      if (metric?.step == null || seen.has(metric.step)) return
      seen.add(metric.step)
      handlers.current.onMetric?.(metric)
    }

    socket.onerror = () => {
      if (disposed) return
      const msg = 'Live metrics connection failed. Training may still be running on the backend.'
      setConnection('error')
      setError(msg)
      handlers.current.onError?.(msg)
    }

    socket.onclose = (event) => {
      if (disposed) return
      setConnection('closed')
      handlers.current.onClose?.(event)
    }

    return () => {
      disposed = true
      socket.onopen = null
      socket.onmessage = null
      socket.onerror = null
      socket.onclose = null
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close()
      }
    }
  }, [experimentId, enabled])

  return { connection, error }
}