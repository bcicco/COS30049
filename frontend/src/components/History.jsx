/** @import { CheckSummary } from '../api/types' */

/** Past checks, newest first. */
export default function History({ /** @type {CheckSummary[]} */ checks, onOpen }) {
  if (checks.length === 0) return null
  return (
    <section className="history">
      <h2>History</h2>
      <ul>
        {checks.map((c) => (
          <li key={c.id}>
            <button type="button" className="link" onClick={() => onOpen(c.id)}>
              <span className="muted">{new Date(c.created_at).toLocaleString()}</span>{' '}
              {Math.round(c.document_score * 100)}% AI · {c.preview}
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
