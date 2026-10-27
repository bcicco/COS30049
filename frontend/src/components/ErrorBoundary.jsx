import { Component } from 'react'
import '../App.css'

/** Catches render errors anywhere below it so the user gets a page, not a blank screen. */
export default class ErrorBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error, info) {
    console.error(error, info.componentStack)
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <main className="not-found" role="alert">
        <h1>Something went wrong</h1>
        <p className="muted">The page hit an unexpected error. Reloading usually fixes it.</p>
        <button type="button" onClick={() => window.location.reload()}>
          Reload
        </button>
      </main>
    )
  }
}
