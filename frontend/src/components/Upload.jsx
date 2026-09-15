import React, { useRef, useState } from 'react';
import { api } from '../api.js';

const LABEL = {
  ok: ['✅ Indexed', 'ok'],
  upload_failed: ['🚫 Upload failed', 'bad'],
  read_failed: ['❌ Read failed', 'bad'],
  empty_text: ['⚠️ No extractable text', 'warn'],
  duplicate: ['♻️ Duplicate', 'warn'],
};

export default function Upload({ onDone }) {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const pick = (list) => setFiles((f) => [...f, ...Array.from(list)]);

  const ingest = async () => {
    setBusy(true);
    setResults([]);
    try {
      const res = await api.upload(files);
      setResults(res.results);
      setFiles([]);
      onDone();
    } catch (e) {
      setResults([{ filename: '(request)', status: 'upload_failed',
                    detail: `Upload request failed: ${e.message}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="main-scroll">
      <h1>Upload circulars</h1>
      <p style={{ color: 'var(--muted)' }}>
        PDF, DOCX, TXT or scanned images (PNG/JPG/TIFF). Scanned PDFs with no text
        layer are read automatically with OCR. Corrupt files, password-protected
        PDFs, unreadable scans and duplicates are flagged here and on the dashboard.
      </p>

      <div
        className={`dropzone ${drag ? 'drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
        onClick={() => inputRef.current.click()}
      >
        <input
          ref={inputRef} type="file" multiple accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.tiff,.tif"
          onChange={(e) => pick(e.target.files)}
        />
        Drop files here or <span className="browse">browse</span>
        {files.length > 0 && (
          <div style={{ marginTop: 10, color: 'var(--text)' }}>
            {files.map((f, i) => <div key={i}>📄 {f.name}</div>)}
          </div>
        )}
      </div>

      <button className="upload-btn" onClick={ingest} disabled={busy || files.length === 0}>
        {busy ? 'Processing…' : `Ingest ${files.length || ''} file(s)`}
      </button>

      {results.length > 0 && <h3 className="section-title">Results</h3>}
      {results.map((r, i) => {
        const [label, cls] = LABEL[r.status] || [r.status, 'warn'];
        return (
          <div key={i} className={`flag ${cls === 'ok' ? '' : cls}`}
               style={cls === 'ok' ? { background: 'var(--accent-soft)',
                 borderLeft: '4px solid var(--accent)' } : {}}>
            <b>{label}</b> — {r.filename}
            <small>
              {r.detail}
              {r.circular_no ? ` · detected as ${r.circular_no}` : ''}
            </small>
          </div>
        );
      })}
    </div>
  );
}
