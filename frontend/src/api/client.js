/** @import { ScoreRequest, ScoreResponse } from './types' */

const MESSAGES = {
  422: 'The text could not be scored, check it is not empty or over 100,000 characters.',
  503: 'The server is busy, try again in a few seconds.',
}

/** @param {ScoreRequest} body @returns {Promise<ScoreResponse>} */
export async function scoreText(body) {
  // the Vite proxy forwards /api to the FastAPI dev server
  const res = await fetch('/api/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({}))
    const message = typeof detail === 'string' ? detail : MESSAGES[res.status]
    throw new Error(message ?? `Request failed: ${res.status}`)
  }
  return res.json()
}
