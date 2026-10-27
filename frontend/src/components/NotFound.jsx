import '../App.css'

/** Shown for any path other than the checker itself. */
export default function NotFound() {
  return (
    <main className="not-found">
      <h1>Page not found</h1>
      <p className="muted">
        <code>{window.location.pathname}</code> doesn't exist.
      </p>
      <a href="/">Back to the checker</a>
    </main>
  )
}
