// Thin client for the FastAPI backend (same origin: nginx / vite proxy forwards /api)
export async function post(endpoint, fields) {
  const fd = new FormData()
  Object.entries(fields).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') fd.append(k, v)
  })
  const res = await fetch(`/api/${endpoint}`, { method: 'POST', body: fd })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`)
  return body
}

export async function getHealth() {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error('backend unreachable')
  return res.json()
}

export async function getSamples() {
  const res = await fetch('/api/samples')
  return res.ok ? res.json() : { ids: [] }
}
