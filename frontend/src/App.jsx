import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API = 'http://localhost:8000'

function App() {
  const [documents, setDocuments] = useState([])
  const [selected, setSelected] = useState(new Set())
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [links, setLinks] = useState({})
  const [expanded, setExpanded] = useState({})
  const [ingesting, setIngesting] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const fetchDocuments = async () => {
    const res = await fetch(`${API}/documents`)
    const data = await res.json()
    setDocuments(data.documents)
    setSelected(new Set(data.documents))
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
          sources: selected.size === documents.length ? null : Array.from(selected)
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
    fetchDocuments()
  }

  const runIngestion = async () => {
    setIngesting(true)
    await fetch(`${API}/ingest`, { method: 'POST' })
    setIngesting(false)
    fetchDocuments()
  }

  return (
    <div className="app">
      <aside className="panel left">
        <h2>Documents</h2>
        <label className="upload-btn">
          Upload PDF / DOCX
          <input type="file" multiple accept=".pdf,.docx" onChange={handleUpload} hidden />
        </label>
        <button className="ingest-btn" onClick={runIngestion} disabled={ingesting}>
          {ingesting ? 'Ingesting…' : 'Run Ingestion'}
        </button>
        <div className="divider" />
        <p className="count">{documents.length} document(s)</p>
        <ul className="doc-list">
          {documents.map(doc => (
            <li key={doc}>
              <label>
                <input
                  type="checkbox"
                  checked={selected.has(doc)}
                  onChange={() => toggleDoc(doc)}
                />
                <span>{doc}</span>
              </label>
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
        {Array.from(selected).filter(d => d.toLowerCase().endsWith('.pdf')).map(doc => (
          <div key={doc} className="link-group">
            <button className="link-header" onClick={() => toggleLinks(doc)}>
              {expanded[doc] ? '▾' : '▸'} {doc}
            </button>
            {expanded[doc] && (
              <div className="link-items">
                {links[doc]?.length ? links[doc].map((l, i) => (
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