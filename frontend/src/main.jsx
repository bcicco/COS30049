import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import NotFound from './components/NotFound.jsx'

// one page app, so a path check is enough instead of a router
const Page = window.location.pathname === '/' ? App : NotFound

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Page />
  </StrictMode>,
)
