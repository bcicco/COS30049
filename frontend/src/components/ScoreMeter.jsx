/** Document score as a hero figure over a 0-100% meter. */
export default function ScoreMeter({ score }) {
  const pct = Math.round(score * 100)
  return (
    <section className="tile">
      <span className="tile-label">Probability this document is AI generated</span>
      <span className="hero">{pct}% AI</span>
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
      <div className="meter-scale" aria-hidden="true">
        <span>Human</span>
        <span>AI</span>
      </div>
    </section>
  )
}
