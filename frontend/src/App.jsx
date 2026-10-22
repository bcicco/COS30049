import { useEffect, useState } from 'react'
import { getCheck, listChecks, logout, me, scoreText } from './api/client'
import AuthForm from './components/AuthForm'
import Editor from './components/Editor'
import History from './components/History'
import SentenceDetail from './components/SentenceDetail'
import './App.css'

function App() {
  // undefined while checking the session cookie, null when logged out
  const [user, setUser] = useState(undefined)

  useEffect(() => {
    me()
      .then(setUser)
      .catch(() => setUser(null))
  }, [])

  if (user === undefined) return null
  return (
    <main>
      <header>
        <h1>aivhuman</h1>
        {user && (
          <button type="button" className="link" onClick={() => logout().finally(() => setUser(null))}>
            {user.username} · Log out
          </button>
        )}
      </header>
      {user ? <Scorer onLoggedOut={() => setUser(null)} /> : <AuthForm onAuth={setUser} />}
    </main>
  )
}

function Scorer({ onLoggedOut }) {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [selected, setSelected] = useState(null)
  const [checks, setChecks] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  // highlights only line up with the exact text that was scored
  const current = result !== null && result.text === text ? result : null

  // every API failure lands here, an expired session sends you back to the login form
  function fail(err) {
    if (err.status === 401) onLoggedOut()
    else setError(err.message)
  }

  useEffect(() => {
    listChecks()
      .then(setChecks)
      .catch((err) => (err.status === 401 ? onLoggedOut() : setError(err.message)))
  }, [onLoggedOut])

  function show(check) {
    setText(check.result.text)
    setResult(check.result)
    setSelected(null)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    // offsets come back for the NFC text, so score and display that same string
    const nfc = text.normalize('NFC')
    setText(nfc)
    setLoading(true)
    setError(null)
    try {
      show(await scoreText({ text: nfc }))
      setChecks(await listChecks())
    } catch (err) {
      fail(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {current && <p className="score">{Math.round(current.document_score * 100)}% AI</p>}
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
            <mark className="human">Human</mark> <mark className="ai">AI</mark>
          </span>
          <button type="submit" disabled={loading || !text.trim()}>
            {loading ? 'Scoring...' : 'Score'}
          </button>
        </div>
      </form>
      {current && selected !== null && <SentenceDetail sentence={current.sentences[selected]} />}
      {error && <p className="error">{error}</p>}
      <History checks={checks} onOpen={(id) => getCheck(id).then(show).catch(fail)} />
    </>
  )
}

export default App
