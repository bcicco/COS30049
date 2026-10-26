// Mirrors backend/src/aivhuman_api/schemas.py. Keep the two in sync.

/**
 * @typedef {object} ScoreRequest
 * @property {string} text 1 to 100,000 characters
 */

/**
 * start/end are Python code point offsets, which drift from JS (UTF-16) string
 * indices on emoji and other astral characters. Convert with Array.from before slicing.
 * @typedef {object} SentenceScore
 * @property {string} text
 * @property {number} start
 * @property {number} end
 * @property {number} score calibrated p(machine), 0 to 1
 * @property {boolean} flagged over the calibrated threshold, use this to highlight
 * @property {boolean} too_short never flagged
 */

/**
 * @typedef {object} ScoreResponse
 * @property {string} text NFC-normalised input
 * @property {SentenceScore[]} sentences
 * @property {number} document_score 0 to 1
 */
