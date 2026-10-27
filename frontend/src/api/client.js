/** @import { Check, ScoreRequest, ScoreResponse } from './types' */

const MESSAGES = {
  422: 'The text could not be scored, check it is not empty or over 100,000 characters.',
  503: 'The server is busy, try again in a few seconds.',
}

// one id per browser tab, the backend files history under it so tabs never see each other's
const SESSION_KEY = 'aivhuman-session'
if (!sessionStorage.getItem(SESSION_KEY)) sessionStorage.setItem(SESSION_KEY, crypto.randomUUID())
const SESSION_ID = sessionStorage.getItem(SESSION_KEY)

// the Vite proxy forwards /api to the FastAPI dev server
async function request(path, init = {}) {
  const res = await fetch(path, {
    ...init,
    headers: { ...init.headers, 'X-Session-Id': SESSION_ID },
  })
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({}))
    const message = typeof detail === 'string' ? detail : MESSAGES[res.status]
    throw new Error(message ?? `Request failed: ${res.status}`)
  }
  return res.json()
}

/** @param {ScoreRequest} body @returns {Promise<ScoreResponse>} */
export function scoreText(body) {
  return request('/api/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** This tab's past checks, newest first. @returns {Promise<Check[]>} */
export function listChecks() {
  return request('/api/checks')
}
