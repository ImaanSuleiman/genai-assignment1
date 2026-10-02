import React, { useEffect, useRef, useState } from 'react'
import { getSamples, post } from '../api'

export function Card({ title, subtitle, children, className = '' }) {
  return (
    <section className={`rounded-2xl bg-white p-5 shadow-card ring-1 ring-slate-100 ${className}`}>
      {title && (
        <header className="mb-3">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  )
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
}

export function ErrorBox({ message }) {
  if (!message) return null
  return <div role="alert" className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">{message}</div>
}

export function RunButton({ loading, disabled, onClick, children = 'Run model' }) {
  return (
    <button
      onClick={onClick}
      disabled={loading || disabled}
      className="inline-flex items-center justify-center gap-2 rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {loading && <Spinner />} {loading ? 'Running…' : children}
    </button>
  )
}

export function ImagePanel({ title, src, caption, download }) {
  return (
    <figure className="flex flex-col items-center gap-2">
      <div className="aspect-square w-full overflow-hidden rounded-xl bg-slate-100 ring-1 ring-slate-200">
        {src ? <img src={src} alt={title} className="pixelated h-full w-full object-contain" /> : (
          <div className="flex h-full items-center justify-center text-xs text-slate-400">no image yet</div>
        )}
      </div>
      <figcaption className="text-center text-xs font-medium text-slate-600">{title}{caption && <span className="block font-normal text-slate-400">{caption}</span>}</figcaption>
      {download && src && (
        <a href={src} download={download} className="text-xs font-semibold text-brand-600 hover:underline">Download PNG</a>
      )}
    </figure>
  )
}

export function Stat({ label, value }) {
  return (
    <div className="rounded-xl bg-slate-50 px-3 py-2 ring-1 ring-slate-100">
      <div className="text-[11px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="text-sm font-semibold text-slate-800">{value}</div>
    </div>
  )
}

export function Bar({ label, value, highlight }) {
  const pct = Math.round(value * 1000) / 10
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className={highlight ? 'font-semibold text-brand-700' : 'text-slate-600'}>{label}</span>
        <span className="tabular-nums text-slate-500">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${highlight ? 'bg-brand-600' : 'bg-slate-300'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div>
      {label && <div className="mb-1 text-xs font-medium text-slate-500">{label}</div>}
      <div className="inline-flex flex-wrap gap-1 rounded-xl bg-slate-100 p-1">
        {options.map((o) => (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${value === o.value ? 'bg-white text-brand-700 shadow' : 'text-slate-500 hover:text-slate-800'}`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

/** Webcam capture -> File (needs https or localhost for getUserMedia). */
export function Webcam({ onCapture }) {
  const video = useRef(null)
  const stream = useRef(null)
  const [on, setOn] = useState(false)
  const [err, setErr] = useState('')

  const stop = () => {
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
    setOn(false)
  }
  const start = async () => {
    setErr('')
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      video.current.srcObject = stream.current
      await video.current.play()
      setOn(true)
    } catch (e) {
      setErr('Camera unavailable: ' + (e.message || e.name) + ' (use http://localhost or https, and allow camera access).')
    }
  }
  const snap = () => {
    const v = video.current
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d').drawImage(v, 0, 0)
    c.toBlob((b) => {
      onCapture(new File([b], 'webcam.png', { type: 'image/png' }))
      stop()
    }, 'image/png')
  }
  useEffect(() => stop, [])
  return (
    <div className="space-y-2">
      <video ref={video} muted playsInline className={`w-full rounded-xl bg-black ${on ? '' : 'hidden'}`} />
      <div className="flex gap-2">
        {!on ? (
          <button onClick={start} className="rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-200">Use webcam</button>
        ) : (
          <>
            <button onClick={snap} className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white">Capture photo</button>
            <button onClick={stop} className="rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700">Cancel</button>
          </>
        )}
      </div>
      {err && <p className="text-xs text-red-600">{err}</p>}
    </div>
  )
}

/** Image source picker: upload, optional clean samples, optional webcam. */
export function ImageSource({ value, onChange, samples = true, webcam = false, accept = 'image/jpeg,image/png,image/webp' }) {
  const [ids, setIds] = useState([])
  const [preview, setPreview] = useState(null)
  const input = useRef(null)

  useEffect(() => { if (samples) getSamples().then((s) => setIds(s.ids || [])).catch(() => {}) }, [samples])
  useEffect(() => {
    if (value?.file) {
      const url = URL.createObjectURL(value.file)
      setPreview(url)
      return () => URL.revokeObjectURL(url)
    }
    setPreview(value?.sampleId !== undefined ? `/api/samples/${value.sampleId}` : null)
  }, [value])

  const drop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f) onChange({ file: f })
  }
  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={drop}
        onClick={() => input.current.click()}
        className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 p-4 text-center hover:border-brand-500"
      >
        {preview ? <img src={preview} alt="selected" className="pixelated h-36 w-36 rounded-lg object-cover" /> : (
          <span className="text-sm text-slate-500">Drop an image here or click to upload<br /><span className="text-xs text-slate-400">JPEG, PNG or WebP, up to 10 MB</span></span>
        )}
        <input ref={input} type="file" accept={accept} hidden onChange={(e) => e.target.files[0] && onChange({ file: e.target.files[0] })} />
      </div>
      {samples && ids.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-medium text-slate-500">or pick a clean sample</div>
          <div className="flex flex-wrap gap-2">
            {ids.map((id) => (
              <button key={id} onClick={() => onChange({ sampleId: id })} className={`overflow-hidden rounded-lg ring-2 ${value?.sampleId === id ? 'ring-brand-600' : 'ring-transparent hover:ring-slate-300'}`}>
                <img src={`/api/samples/${id}`} alt={`sample ${id}`} className="pixelated h-12 w-12 object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
      {webcam && <Webcam onCapture={(file) => onChange({ file })} />}
    </div>
  )
}

export function useInference(endpoint) {
  const [state, setState] = useState({ loading: false, error: '', result: null })
  const run = async (fields) => {
    setState((s) => ({ ...s, loading: true, error: '' }))
    try {
      const result = await post(endpoint, fields)
      setState({ loading: false, error: '', result })
    } catch (e) {
      setState((s) => ({ ...s, loading: false, error: e.message }))
    }
  }
  return { ...state, run }
}

export const sourceFields = (v) => (v?.file ? { file: v.file } : v?.sampleId !== undefined ? { sample_id: v.sampleId } : {})

export const CORRUPTIONS = [
  { value: 'already_corrupted', label: 'Image is already corrupted' },
  { value: 'salt', label: 'Salt-and-pepper' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Occlusion' },
  { value: 'none', label: 'None (clean)' },
]
export const LEVELS = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]
export const fmtMs = (v) => `${v.toFixed(1)} ms`
