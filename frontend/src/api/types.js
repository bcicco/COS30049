// Mirrors backend/src/aivhuman_api/schemas.py. Keep the two in sync.

/** Mirrors aivhuman.mil.infer.MAX_CHARS. */
export const MAX_CHARS = 100_000

/**
 * @typedef {object} ScoreRequest
 * @property {string} text 1 to MAX_CHARS characters
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

/**
 * @typedef {object} Credentials
 * @property {string} username 3 to 32 of A-Z a-z 0-9 _ -
 * @property {string} password 8 to 128 characters
 */

/**
 * @typedef {object} Me
 * @property {string} username
 */

/**
 * @typedef {object} CheckSummary
 * @property {number} id
 * @property {string} created_at ISO 8601, UTC
 * @property {number} document_score
 * @property {string} preview first 100 characters of the text
 */

/**
 * @typedef {object} CheckOut
 * @property {number} id
 * @property {string} created_at ISO 8601, UTC
 * @property {ScoreResponse} result
 */
