import { useState } from 'react'

/** Document score as a hero figure over a 0-100% meter. */
export default function ScoreMeter({ score }) {
  const [hover, setHover] = useState(false)
  const pct = Math.round(score * 100)
  return (
    <section className="tile">
      <span className="tile-label">Probability this document is AI generated</span>
      <span className="hero">{pct}% AI</span>
      <div
        className="meter-wrap"
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
        <div
          className="meter"
          role="meter"
          aria-label="Likelihood the text is AI-written"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-valuetext={`${pct}% AI`}
        >
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
        {/* follows the fill end, clamped so it never pokes out of the tile */}
        {hover && (
          <div className="tooltip" role="presentation" style={{ left: `${Math.min(Math.max(pct, 15), 85)}%` }}>
            <strong>{pct}% AI</strong>
            <span>{100 - pct}% human</span>
            <span className="muted">Whole-document score</span>
          </div>
        )}
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>Human</span>
        <span>AI</span>
      </div>
    </section>
  )
}
