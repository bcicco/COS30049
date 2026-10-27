/** @import { SentenceScore } from '../api/types' */
import { useState } from 'react'

const BINS = 10
// Stacked bottom to top in this order, validated for colour-blind separation.
const KINDS = [
  { key: 'ai', label: 'AI' },
  { key: 'human', label: 'Human' },
  { key: 'short', label: 'Too short' },
]

/** @param {SentenceScore} s */
function kindOf(s) {
  if (s.too_short) return 'short'
  return s.flagged ? 'ai' : 'human'
}

function binLabel(i) {
  return `${i * 10}-${(i + 1) * 10}%`
}

function describe(bin) {
  return KINDS.filter((k) => bin[k.key] > 0)
    .map((k) => `${bin[k.key]} ${k.label.toLowerCase()}`)
    .join(', ')
}

/** How many sentences fall in each 10% score band, split by how they were highlighted. */
export default function ScoreHistogram({ /** @type {SentenceScore[]} */ sentences }) {
  const [hover, setHover] = useState(null)

  const bins = Array.from({ length: BINS }, () => ({ ai: 0, human: 0, short: 0, total: 0 }))
  for (const s of sentences) {
    const bin = bins[Math.min(Math.floor(s.score * BINS), BINS - 1)]
    bin[kindOf(s)]++
    bin.total++
  }
  const max = Math.max(...bins.map((b) => b.total))

  return (
    <section className="tile">
      <div className="tile-header">
        <span className="tile-label">Sentence scores</span>
        <ul className="legend">
          {KINDS.map((k) => (
            <li key={k.key}>
              <span className={`swatch ${k.key}`} />
              {k.label}
            </li>
          ))}
        </ul>
      </div>
      <div className="plot">
        <div className="y-axis" aria-hidden="true">
          <span>{max}</span>
          <span>0</span>
        </div>
        <div className="bars">
          {bins.map((bin, i) => (
            <div
              key={i}
              className="bin"
              role="img"
              aria-label={`${binLabel(i)} AI: ${bin.total} sentences${bin.total ? ` (${describe(bin)})` : ''}`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              <div className="stack" style={{ height: `${(bin.total / max) * 100}%` }}>
                {KINDS.map(
                  (k) =>
                    bin[k.key] > 0 && (
                      <div key={k.key} className={`seg ${k.key}`} style={{ flexGrow: bin[k.key] }} />
                    ),
                )}
              </div>
              {hover === i && (
                <div className="tooltip" role="presentation">
                  <strong>{binLabel(i)} AI</strong>
                  <span>
                    {bin.total} {bin.total === 1 ? 'sentence' : 'sentences'}
                  </span>
                  {bin.total > 0 && <span className="muted">{describe(bin)}</span>}
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="x-axis" aria-hidden="true">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    </section>
  )
}
