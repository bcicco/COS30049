import { useState } from 'react'
import { login, register } from '../api/client'

/** Log in or create an account. */
export default function AuthForm({ onAuth }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const isLogin = mode === 'login'

  async function submit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      onAuth(await (isLogin ? login : register)({ username, password }))
    } catch (err) {
      setError(err.status === 422 ? 'Username: 3-32 letters, digits, _ or -. Password: 8+ characters.' : err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form className="auth" onSubmit={submit}>
      <input
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        placeholder="Username"
        autoComplete="username"
        required
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        autoComplete={isLogin ? 'current-password' : 'new-password'}
        required
      />
      <button type="submit" disabled={loading}>
        {isLogin ? 'Log in' : 'Create account'}
      </button>
      {error && <p className="error">{error}</p>}
      <button type="button" className="link" onClick={() => setMode(isLogin ? 'register' : 'login')}>
        {isLogin ? 'No account? Create one' : 'Have an account? Log in'}
      </button>
    </form>
  )
}
