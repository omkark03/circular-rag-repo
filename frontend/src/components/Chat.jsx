import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '../api.js';

function RefCard({ r, onPreview, relevance }) {
  return (
    <div className="ref-card" role="button" tabIndex={0}
         onClick={() => onPreview({ name: r.title, url: r.url })}>
      {r.circular_no && <div className="circ-no">CIRCULAR NO. {r.circular_no}</div>}
      {r.issuer}. ({r.year}). <i>{r.title}</i>.
      <div>
        <em>Click to preview · </em>
        <a href={r.url} target="_blank" rel="noreferrer"
           onClick={(e) => e.stopPropagation()}>open in new tab ↗</a>
        {relevance != null && (
          <span className="relevance-tag" title="Retrieval relevance score for this question">
            {' '}· relevance {relevance}
          </span>
        )}
      </div>
    </div>
  );
}

function ContradictionsPanel({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="contradictions-panel">
      <div className="contradictions-head">
        ⚠️ Conflicting circulars found — review before relying on this answer
      </div>
      {items.map((c, i) => (
        <div key={i} className="contradiction-item">
          <div className="contradiction-topic">{c.topic}</div>
          {c.positions.map((p, j) => (
            <div key={j} className="contradiction-position">
              <span className="circ-no">{p.issuer} ({p.year})</span>
              <span className="contradiction-claim">{p.claim}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function Message({ m, onPreview }) {
  // Best relevance score seen for each source filename, so a reference
  // card's number is visually traceable to what actually matched.
  const bestScore = {};
  for (const h of m.hits || []) {
    if (bestScore[h.filename] === undefined || h.score > bestScore[h.filename]) {
      bestScore[h.filename] = h.score;
    }
  }
  return (
    <div className={`msg ${m.role}`}>
      <div className="who">{m.role === 'user' ? 'You' : 'Repository assistant'}</div>
      <div className="bubble">
        {m.content
          ? (m.role === 'assistant'
              ? (
                <div className="md">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({ node, ...props }) => (
                        <div className="md-table-wrap"><table {...props} /></div>
                      ),
                    }}
                  >
                    {m.content}
                  </ReactMarkdown>
                </div>
              )
              : m.content)
          : (m.streaming ? <span className="thinking">Retrieving circulars…</span> : null)}
        {m.translating && (
          <div className="thinking" style={{ marginTop: 6 }}>Translating to Marathi…</div>
        )}
        {m.translationNote && (
          <div className="translation-note">⚠️ {m.translationNote}</div>
        )}
        {m.extractingSchedule && (
          <div className="thinking" style={{ marginTop: 6 }}>
            Reading station names and date from your message…
          </div>
        )}
        {m.checkingDurationRules && (
          <div className="thinking" style={{ marginTop: 6 }}>
            Checking circulars for inspection duration rules…
          </div>
        )}
        {m.checkingContradictions && (
          <div className="thinking" style={{ marginTop: 6 }}>
            Checking for conflicts between circulars…
          </div>
        )}
        <ContradictionsPanel items={m.contradictions} />
        {m.references?.length > 0 && (
          <div className="ref-list">
            <div className="who" style={{ marginTop: 8 }}>References (APA)</div>
            {m.references.map((r, i) => (
              <RefCard key={i} r={r} onPreview={onPreview}
                       relevance={bestScore[r.filename] ?? null} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Chat({ chatId, onFirstMessage, onNeedChat }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState(null);
  const bottom = useRef(null);

  useEffect(() => {
    if (chatId) api.getChat(chatId).then((d) => setMessages(d.messages));
    else setMessages([]);
  }, [chatId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    if (!chatId) { onNeedChat(); return; }
    setInput('');
    setBusy(true);
    setMessages((m) => [
      ...m,
      { role: 'user', content: q, references: [] },
      { role: 'assistant', content: '', references: [], hits: [], streaming: true },
    ]);
    const patchLast = (fn) => setMessages((m) => {
      const copy = [...m];
      copy[copy.length - 1] = fn(copy[copy.length - 1]);
      return copy;
    });
    try {
      await api.askStream(chatId, q, {
        onRefs: (refs) => patchLast((last) => ({ ...last, references: refs })),
        onHits: (hits) => patchLast((last) => ({ ...last, hits })),
        onToken: (t) => patchLast((last) => ({ ...last, content: last.content + t })),
        onTranslating: () => patchLast((last) => ({ ...last, translating: true })),
        onTranslation: (text) => patchLast((last) => ({
          ...last, translating: false, contentEn: last.content, content: text,
        })),
        onTranslationUnavailable: (detail) => patchLast((last) => ({
          ...last, translating: false, translationNote: detail,
        })),
        onCheckingContradictions: () => patchLast((last) => ({
          ...last, checkingContradictions: true,
        })),
        onContradictions: (items) => patchLast((last) => ({
          ...last, checkingContradictions: false, contradictions: items,
        })),
        onExtractingSchedule: () => patchLast((last) => ({
          ...last, extractingSchedule: true,
        })),
        onCheckingDurationRules: () => patchLast((last) => ({
          ...last, extractingSchedule: false, checkingDurationRules: true,
        })),
      });
      patchLast((last) => ({
        ...last, streaming: false, checkingContradictions: false,
        extractingSchedule: false, checkingDurationRules: false,
      }));
      onFirstMessage(); // refresh sidebar titles
    } catch (e) {
      patchLast((last) => ({
        ...last, streaming: false,
        content: last.content ||
          `⚠️ Request failed: ${e.message}. Is the backend running on port 8000?`,
      }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="main-scroll">
        {!chatId && (
          <div className="msg">
            <h2>Ask the circular repository</h2>
            <p style={{ color: 'var(--muted)' }}>
              Answers are generated locally, strictly from your uploaded circulars,
              with APA-style clickable references. Start a new chat from the sidebar,
              or just type below.
            </p>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} m={m} onPreview={setPreview} />)}
        <div ref={bottom} />
      </div>
      {preview && (
        <div className="modal-scrim" onClick={() => setPreview(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <span className="modal-title">{preview.name}</span>
              <a href={preview.url} target="_blank" rel="noreferrer">Open in new tab ↗</a>
              <button className="modal-close" onClick={() => setPreview(null)}>✕</button>
            </div>
            <iframe title="Circular preview" src={preview.url} className="modal-frame" />
          </div>
        </div>
      )}
      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="e.g. What is the KYC requirement for small accounts?"
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim()}>Ask</button>
      </div>
    </>
  );
}
