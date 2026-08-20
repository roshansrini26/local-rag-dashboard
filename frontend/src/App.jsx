import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API = 'http://localhost:8000'

function App() {
  const [sources, setSources] = useState([])
  const [selected, setSelected] = useState(new Set())
  const [urlInput, setUrlInput] = useState('')
  const [checking, setChecking] = useState(false)
  const [addingUrl, setAddingUrl] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [links, setLinks] = useState({})
  const [expanded, setExpanded] = useState({})
  const [ingesting, setIngesting] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    fetchSources()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const fetchSources = async () => {
    const res = await fetch(`${API}/sources`)
    const data = await res.json()
    setSources(data.sources)
    setSelected(new Set(data.sources.map(s => s.id)))
  }

  const toggleDoc = (name) => {
    const next = new Set(selected)
    next.has(name) ? next.delete(name) : next.add(name)
    setSelected(next)
  }

  const toggleLinks = async (name) => {
    setExpanded(prev => ({ ...prev, [name]: !prev[name] }))
    if (!links[name]) {
      const res = await fetch(`${API}/links/${encodeURIComponent(name)}`)
      const data = await res.json()
      setLinks(prev => ({ ...prev, [name]: data.links }))
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const question = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          sources: selected.size === sources.length ? null : Array.from(selected)
        })
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: could not reach the API.' }])
    }
    setLoading(false)
  }

  const handleUpload = async (e) => {
    const files = e.target.files
    if (!files.length) return
    const formData = new FormData()
    for (const f of files) formData.append('files', f)
    await fetch(`${API}/upload`, { method: 'POST', body: formData })
    fetchSources()
  }

  const runIngestion = async () => {
    setIngesting(true)
    await fetch(`${API}/ingest`, { method: 'POST' })
    setIngesting(false)
    fetchSources()
  }

    const addUrl = async () => {
    if (!urlInput.trim()) return
    setAddingUrl(true)
    await fetch(`${API}/add-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: urlInput.trim() })
    })
    setUrlInput('')
    setAddingUrl(false)
    fetchSources()
  }

  const checkUpdates = async () => {
    setChecking(true)
    const res = await fetch(`${API}/check-updates`, { method: 'POST' })
    const data = await res.json()
    setChecking(false)
    fetchSources()
    if (data.count === 0) alert('All sources up to date.')
  }

  const approveUpdate = async (url) => {
    await fetch(`${API}/approve-update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    })
    fetchSources()
  }

  const rejectUpdate = async (url) => {
    await fetch(`${API}/reject-update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    })
    fetchSources()
  }

  return (
    <div className="app">
      <aside className="panel left">
        <h2>Sources</h2>

        <label className="upload-btn">
          Upload PDF / DOCX
          <input type="file" multiple accept=".pdf,.docx" onChange={handleUpload} hidden />
        </label>
        <button className="ingest-btn" onClick={runIngestion} disabled={ingesting}>
          {ingesting ? 'Ingesting…' : 'Run Ingestion'}
        </button>

        <div className="divider" />

        <input
          className="url-input"
          type="text"
          placeholder="https://regulation-page…"
          value={urlInput}
          onChange={e => setUrlInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addUrl()}
        />
        <button className="ingest-btn" onClick={addUrl} disabled={addingUrl || !urlInput.trim()}>
          {addingUrl ? 'Adding…' : 'Track URL'}
        </button>
        <button className="ingest-btn" onClick={checkUpdates} disabled={checking}>
          {checking ? 'Checking…' : 'Check for updates'}
        </button>

        <div className="divider" />

        <p className="count">{sources.length} source(s)</p>
        <ul className="doc-list">
          {sources.map(s => (
            <li key={s.id} className={s.status === 'changed' ? 'changed' : ''}>
              <label>
                <input
                  type="checkbox"
                  checked={selected.has(s.id)}
                  onChange={() => toggleDoc(s.id)}
                />
                <span>
                  <span className="type-tag">{s.type === 'url' ? 'URL' : 'DOC'}</span>
                  {s.type === 'url' ? new URL(s.url).hostname : s.label}
                </span>
              </label>
              {s.status === 'changed' && (
                <div className="change-actions">
                  <span className="badge">Source changed</span>
                  <button onClick={() => approveUpdate(s.url)}>Approve</button>
                  <button onClick={() => rejectUpdate(s.url)}>Reject</button>
                </div>
              )}
            </li>
          ))}
        </ul>
      </aside>

      <main className="panel center">
        <h2>Chat</h2>
        <div className="chat-scroll">
          {messages.length === 0 && (
            <p className="empty">Ask a question about your documents.</p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">
                {m.role === 'assistant'
                  ? <ReactMarkdown>{m.content}</ReactMarkdown>
                  : m.content}
              </div>
            </div>
          ))}
          {loading && <div className="msg assistant"><div className="bubble thinking">Thinking…</div></div>}
          <div ref={chatEndRef} />
        </div>
        <div className="input-bar">
          <input
            type="text"
            value={input}
            placeholder="Ask a question about your documents…"
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>Send</button>
        </div>
      </main>

      <aside className="panel right">
        <h2>Quick Links</h2>
        {sources
          .filter(s => selected.has(s.id) && s.type === 'file' && s.label.toLowerCase().endsWith('.pdf'))
          .map(s => (
            <div key={s.id} className="link-group">
              <button className="link-header" onClick={() => toggleLinks(s.label)}>
                {expanded[s.label] ? '▾' : '▸'} {s.label}
              </button>
              {expanded[s.label] && (
                <div className="link-items">
                  {links[s.label]?.length ? links[s.label].map((l, i) => (
                    <a key={i} href={l.url} target="_blank" rel="noreferrer">
                      {l.url} <span className="page">p.{l.page}</span>
                    </a>
                  )) : <p className="empty small">No links found.</p>}
                </div>
              )}
            </div>
          ))}
      </aside>
    </div>
  )
}

export default App