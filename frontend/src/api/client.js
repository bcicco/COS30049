/** @import { CheckOut, CheckSummary, Credentials, Me, ScoreRequest } from './types' */

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

const MESSAGES = {
  401: 'Please log in again.',
  503: 'The server is busy, try again in a few seconds.',
}

// The session cookie rides along automatically: the Vite proxy makes /api same-origin.
async function request(path, { method = 'GET', body } = {}) {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({}))
    const message = typeof detail === 'string' ? detail : MESSAGES[res.status]
    throw new ApiError(res.status, message ?? `Request failed: ${res.status}`)
  }
  return res.status === 204 ? null : res.json()
}

/** @returns {Promise<Me>} */
export const me = () => request('/auth/me')
/** @param {Credentials} creds @returns {Promise<Me>} */
export const register = (creds) => request('/auth/register', { method: 'POST', body: creds })
/** @param {Credentials} creds @returns {Promise<Me>} */
export const login = (creds) => request('/auth/login', { method: 'POST', body: creds })
export const logout = () => request('/auth/logout', { method: 'POST' })

/** @param {ScoreRequest} body @returns {Promise<CheckOut>} */
export const scoreText = (body) => request('/score', { method: 'POST', body })
/** @returns {Promise<CheckSummary[]>} */
export const listChecks = () => request('/checks')
/** @param {number} id @returns {Promise<CheckOut>} */
export const getCheck = (id) => request(`/checks/${id}`)
