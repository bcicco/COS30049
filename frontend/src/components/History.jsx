/** @import { Check } from '../api/types' */

/** Checks made in this tab, newest first. */
export default function History({ /** @type {Check[]} */ checks }) {
  if (checks.length === 0) return null
  return (
    <section className="tile">
      <span className="tile-label">Recent checks</span>
      <ul className="history">
        {checks.map((c) => (
          <li key={c.id}>
            <span className="history-time">
              {new Date(c.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <span className="history-text">{c.text}</span>
            <strong>{Math.round(c.document_score * 100)}% AI</strong>
          </li>
        ))}
      </ul>
    </section>
  )
}
