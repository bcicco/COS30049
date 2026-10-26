// Mirrors backend/src/aivhuman_api/schemas.py. Keep the two in sync.

/**
 * @typedef {object} ScoreRequest
 * @property {string} text 1 to 100,000 characters
 */

/**
 * Render `text` directly: start/end are Python code point offsets, which drift
 * from JS (UTF-16) string indices on emoji and other astral characters.
 * @typedef {object} SentenceScore
 * @property {string} text
 * @property {number} start
 * @property {number} end
 * @property {number} n_tokens
 * @property {number} score calibrated p(machine), 0 to 1
 * @property {boolean} flagged
 * @property {boolean} too_short never flagged
 */

/**
 * @typedef {object} ScoreResponse
 * @property {string} model run name, e.g. "p5-adv"
 * @property {string} text NFC-normalised input
 * @property {SentenceScore[]} sentences
 * @property {number} document_score 0 to 1
 * @property {boolean} document_flagged
 */
