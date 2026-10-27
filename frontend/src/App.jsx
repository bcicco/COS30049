import { useState } from "react";
import { scoreText } from "./api/client";
import "./App.css";
import Editor from "./components/Editor";
import ScoreHistogram from "./components/ScoreHistogram";
import ScoreMeter from "./components/ScoreMeter";
import SentenceDetail from "./components/SentenceDetail";

// mirrors MAX_CHARS in backend/src/aivhuman_api/schemas.py
const MAX_CHARS = 100_000;

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  // highlights only line up with the exact text that was scored
  const current = result !== null && result.text === text ? result : null;
  // backend counts code points, text.length would count emoji twice
  const chars = Array.from(text).length;
  const tooLong = chars > MAX_CHARS;

  async function handleSubmit(e) {
    e.preventDefault();
    if (!text.trim() || tooLong) return;
    // offsets come back for the NFC text, so score and display that same string
    const nfc = text.normalize("NFC");
    setText(nfc);
    setLoading(true);
    setError(null);
    try {
      const res = await scoreText({ text: nfc });
      setText(res.text);
      setResult(res);
      setSelected(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>COS30049 Assignment 3</h1>
      {current && (
        <div className="charts">
          <ScoreMeter score={current.document_score} />
          <ScoreHistogram sentences={current.sentences} />
        </div>
      )}
      <form onSubmit={handleSubmit}>
        <Editor
          text={text}
          onChange={setText}
          result={current}
          selected={selected}
          onSelect={setSelected}
        />
        <div className="footer">
          <span>
            <mark className="human">Human</mark> <mark className="ai">AI</mark> <mark className="short">Too short</mark>
          </span>
          <span className="submit">
            <span className={tooLong ? "count error" : "count"}>
              {chars.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
            <button type="submit" disabled={loading || !text.trim() || tooLong}>
              {loading ? "Scoring..." : "Score"}
            </button>
          </span>
        </div>
      </form>
      {current && selected !== null && (
        <SentenceDetail sentence={current.sentences[selected]} />
      )}
      {tooLong && (
        <p className="error">
          Text is {chars.toLocaleString()} characters, the limit is{" "}
          {MAX_CHARS.toLocaleString()}.
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </main>
  );
}

export default App;
