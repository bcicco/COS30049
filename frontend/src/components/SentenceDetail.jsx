/** @import { SentenceScore } from '../api/types' */

/** Score for the clicked sentence. */
export default function SentenceDetail({ /** @type {SentenceScore} */ sentence }) {
  return (
    <p className="detail">
      <strong>{Math.round(sentence.score * 100)}% AI</strong>
      {sentence.too_short && ' (too short to be reliable)'}
    </p>
  )
}
