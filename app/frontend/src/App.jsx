import React, { useEffect, useState } from 'react'
import { getHealth } from './api'
import Restoration from './workspaces/Restoration'
import FaceToSketch from './workspaces/FaceToSketch'

const TABS = [
  { id: 'universal', label: 'Universal Restoration', task: 'Task 1', el: <Restoration variant="universal" /> },
  { id: 'hard', label: 'Hard-Routed Restoration', task: 'Task 2', el: <Restoration variant="hard" /> },
  { id: 'soft', label: 'Soft Mixture-of-Experts', task: 'Task 3', el: <Restoration variant="soft" /> },
  { id: 'sketch', label: 'Face-to-Sketch Generator', task: 'Task 4', el: <FaceToSketch /> },
]

function Health() {
  const [h, setH] = useState(null)
  useEffect(() => {
    const load = () => getHealth().then(setH).catch(() => setH({ status: 'down' }))
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])
  const missing = h?.models ? Object.values(h.models).filter((m) => !m.available).length : 0
  const ok = h?.status === 'ok' && missing === 0
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500" title={h?.models ? JSON.stringify(h.models, null, 1) : ''}>
      <span className={`h-2.5 w-2.5 rounded-full ${ok ? 'bg-emerald-500' : h?.status === 'ok' ? 'bg-amber-500' : 'bg-red-500'}`} />
      {h == null ? 'connecting…' : h.status !== 'ok' ? 'backend offline' : missing ? `${missing} model file(s) missing` : 'backend ready · all models found'}
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState(() => localStorage.getItem('tab') || 'universal')
  useEffect(() => { localStorage.setItem('tab', tab) }, [tab])
  const current = TABS.find((t) => t.id === tab) || TABS[0]
  return (
    <div className="min-h-screen lg:flex">
      <aside className="border-b border-slate-200 bg-white lg:min-h-screen lg:w-72 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="p-5">
          <div className="text-lg font-extrabold tracking-tight text-brand-700">Restoration &amp; Sketch Studio</div>
          <p className="mt-1 text-xs text-slate-400">Generative AI · Assignment 1 · four models, one app</p>
        </div>
        <nav className="flex gap-2 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`shrink-0 rounded-xl px-4 py-3 text-left transition ${tab === t.id ? 'bg-brand-50 ring-1 ring-brand-100' : 'hover:bg-slate-50'}`}
            >
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{t.task}</div>
              <div className={`text-sm font-semibold ${tab === t.id ? 'text-brand-700' : 'text-slate-700'}`}>{t.label}</div>
            </button>
          ))}
        </nav>
        <div className="hidden p-5 lg:block"><Health /></div>
      </aside>
      <main className="min-w-0 flex-1 p-4 sm:p-8">
        <div className="mb-4 lg:hidden"><Health /></div>
        <div key={current.id}>{current.el}</div>
      </main>
    </div>
  )
}
