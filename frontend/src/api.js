const EXPLICIT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL

const BACKEND_TARGET = (import.meta.env.VITE_BACKEND_TARGET || 'chroma').toLowerCase()
const CHROMA_API_BASE_URL = import.meta.env.VITE_CHROMA_API_BASE_URL || 'http://localhost:8000'
const PINECONE_API_BASE_URL = import.meta.env.VITE_PINECONE_API_BASE_URL || 'http://localhost:8001'

const API_BASE_URL =
  EXPLICIT_API_BASE_URL || (BACKEND_TARGET === 'pinecone' ? PINECONE_API_BASE_URL : CHROMA_API_BASE_URL)

export async function generateResponse({ query, mode, temperature, max_tokens }) {
  const res = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      mode,
      temperature,
      max_tokens
    })
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed (${res.status})`)
  }

  return res.json()
}
