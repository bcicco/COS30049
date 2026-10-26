/** @import { ScoreResponse } from '../api/types' */
import { Fragment, useLayoutEffect, useRef } from 'react'

/**
 * The input textarea with sentence highlights drawn on a backdrop behind it.
 * Both layers share font, padding and wrapping, so the marks line up with the text.
 */
export default function Editor({ text, onChange, result, selected, onSelect }) {
  const ref = useRef(null)

  // Grow with the content so the textarea never scrolls out of step with the backdrop.
  useLayoutEffect(() => {
    const el = ref.current
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text])

  // The textarea takes the click, so find the sentence under the caret it leaves behind.
  function handleClick(e) {
    if (!result) return
    // caret is a UTF-16 index, sentence offsets are code points
    const cp = Array.from(text.slice(0, e.target.selectionStart)).length
    const i = result.sentences.findIndex((s) => s.start <= cp && cp < s.end)
    onSelect(i === -1 ? null : i)
  }

  return (
    <div className={result ? 'editor scored' : 'editor'}>
      {result && <Backdrop result={result} selected={selected} />}
      <textarea
        ref={ref}
        className="editor-layer"
        value={text}
        onChange={(e) => onChange(e.target.value)}
        onClick={handleClick}
        placeholder="Paste text to check..."
        aria-label="Text to score"
        spellCheck={false}
      />
    </div>
  )
}

function Backdrop({ /** @type {ScoreResponse} */ result, selected }) {
  // Offsets are code points, Array.from splits by code point so gaps line up with them.
  const chars = Array.from(result.text)
  const { sentences } = result
  return (
    <div className="editor-layer backdrop" aria-hidden="true">
      {sentences.map((s, i) => (
        <Fragment key={s.start}>
          {chars.slice(i === 0 ? 0 : sentences[i - 1].end, s.start).join('')}
          <mark className={`${s.flagged ? 'ai' : 'human'}${i === selected ? ' selected' : ''}`}>
            {s.text}
          </mark>
        </Fragment>
      ))}
      {chars.slice(sentences.at(-1)?.end ?? 0).join('')}
      {/* textarea keeps a line for a trailing newline, the div needs a char to match */}
      {result.text.endsWith('\n') && ' '}
    </div>
  )
}
