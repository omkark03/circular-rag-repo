import React, { useCallback, useEffect, useState } from 'react';
import { api } from '../api.js';

const PILL = {
  ok: ['Indexed', 'ok'],
  upload_failed: ['Upload failed', 'bad'],
  read_failed: ['Read failed', 'bad'],
  empty_text: ['No text', 'warn'],
  duplicate: ['Duplicate', 'warn'],
  ocr_suspect: ['OCR language mismatch', 'bad'],
};

function CircularNoCell({ doc, editable, onSaved }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(doc.circular_no || '');
  const [busy, setBusy] = useState(false);

  if (!editing) {
    return (
      <td style={{ fontFamily: 'var(--mono)', fontSize: '.74rem' }}>
        {doc.circular_no || '—'}
        {editable && (
          <button className="mini-edit" title="Set/correct circular number"
                  onClick={() => setEditing(true)}>✎</button>
        )}
      </td>
    );
  }
  const save = async () => {
    setBusy(true);
    try {
      await api.setCircularNo(doc.id, value.trim());
      setEditing(false);
      onSaved();
    } finally {
      setBusy(false);
    }
  };
  return (
    <td>
      <div className="inline-edit">
        <input value={value} autoFocus
               onChange={(e) => setValue(e.target.value)}
               onKeyDown={(e) => e.key === 'Enter' && save()}
               placeholder="e.g. 10/15" />
        <button onClick={save} disabled={busy}>{busy ? '…' : '✓'}</button>
        <button onClick={() => setEditing(false)} disabled={busy}>✕</button>
      </div>
    </td>
  );
}

export default function Dashboard({ role, onChanged }) {
  const [data, setData] = useState(null);
  const [recheckMsg, setRecheckMsg] = useState('');
  const [rechecking, setRechecking] = useState(false);
  const [preview, setPreview] = useState(null); // {name, url}
  const [repoSearch, setRepoSearch] = useState('');
  const [selected, setSelected] = useState(() => new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);

  const load = useCallback(() => api.dashboard().then(setData), []);
  useEffect(() => { load(); }, [load]);

  if (!data) return <div className="main-scroll">Loading dashboard…</div>;
  const { stats, failed, missing_references, documents, cross_reference_check_enabled } = data;
  const isAdmin = role === 'admin';

  const q = repoSearch.trim().toLowerCase();
  const filteredDocs = q
    ? documents.filter((d) =>
        [d.filename, d.circular_no, d.issuer, d.title, d.status]
          .some((v) => (v || '').toLowerCase().includes(q)))
    : documents;

  const toggleOne = (id) => setSelected((s) => {
    const next = new Set(s);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });
  const allVisibleSelected = filteredDocs.length > 0
    && filteredDocs.every((d) => selected.has(d.id));
  const toggleAllVisible = () => setSelected((s) => {
    if (allVisibleSelected) {
      const next = new Set(s);
      filteredDocs.forEach((d) => next.delete(d.id));
      return next;
    }
    return new Set([...s, ...filteredDocs.map((d) => d.id)]);
  });

  const bulkDelete = async () => {
    const ids = [...selected];
    if (ids.length === 0) return;
    if (!window.confirm(
      `Delete ${ids.length} selected circular(s) from the repository?\n` +
      'They will no longer be searchable, and other circulars citing them ' +
      'will be flagged as missing references.')) return;
    setBulkDeleting(true);
    try {
      await api.bulkDeleteDocuments(ids);
      setSelected(new Set());
      await load();
      onChanged && onChanged();
    } finally {
      setBulkDeleting(false);
    }
  };

  const recheck = async () => {
    setRechecking(true);
    setRecheckMsg('');
    try {
      const r = await api.recheck();
      setRecheckMsg(r.resolved > 0
        ? `✅ Resolved ${r.resolved} reference(s).`
        : 'No new matches found — the referenced circulars are still not in the repository.');
      await load();
      onChanged && onChanged();
    } catch (e) {
      setRecheckMsg(`⚠️ Re-check failed: ${e.message}`);
    } finally {
      setRechecking(false);
    }
  };

  const removeDoc = async (doc) => {
    if (!window.confirm(
      `Delete "${doc.filename}" from the repository?\n` +
      'It will no longer be searchable, and circulars citing it will be ' +
      'flagged as missing references.')) return;
    await api.deleteDocument(doc.id);
    setSelected((s) => { const next = new Set(s); next.delete(doc.id); return next; });
    await load();
    onChanged && onChanged();
  };

  const onCircularNoSaved = async () => {
    await load();
    onChanged && onChanged();
  };

  return (
    <div className="main-scroll">
      <h1>Dashboard</h1>

      <div className="metrics">
        <div className="metric"><div className="n">{stats.total}</div>
          <div className="l">Total uploads</div></div>
        <div className="metric"><div className="n">{stats.ok}</div>
          <div className="l">Indexed OK</div></div>
        <div className="metric"><div className={`n ${stats.failed ? 'bad' : ''}`}>{stats.failed}</div>
          <div className="l">Upload / read failures</div></div>
        
      </div>

      <h3>🚩 Upload &amp; read failures</h3>
      {failed.length === 0 ? (
        <div className="ok-note">No failed uploads — all files were read and indexed successfully.</div>
      ) : (
        <div className="scroll-window">
          {failed.map((d) => (
            <div key={d.id} className="flag bad">
              <b>{(PILL[d.status] || [d.status])[0]}</b> — {d.filename}
              <small>{d.status_detail} · {d.uploaded_at}</small>
            </div>
          ))}
        </div>
      )}

      {cross_reference_check_enabled && (
        <>
          <h3 className="section-title">🔗 Cross-reference check</h3>
          <p style={{ color: 'var(--muted)', marginTop: 0 }}>
            Circular numbers cited in document bodies that are not present in the repository.
          </p>
          {missing_references.length === 0 ? (
            <div className="ok-note">All circulars referenced in document bodies are available in the repository.</div>
          ) : (
            <>
              <div className="scroll-window">
                {missing_references.map((r) => (
                  <div key={r.id} className="flag warn">
                    <b>Missing: {r.ref_text}</b>
                    <small>
                      Referenced in {r.source_file}
                      {r.source_circular ? ` (Circular ${r.source_circular})` : ''} —
                      upload this circular, or correct its detected number below, to resolve.
                    </small>
                  </div>
                ))}
              </div>
              {role !== 'viewer' && (
                <button className="upload-btn" onClick={recheck} disabled={rechecking}>
                  {rechecking ? 'Checking…' : 'Re-check after new uploads'}
                </button>
              )}
              {recheckMsg && <p style={{ fontSize: '.84rem' }}>{recheckMsg}</p>}
            </>
          )}
        </>
      )}

      <h3 className="section-title">📚 Circulars</h3>
      {documents.length === 0 ? (
        <div className="ok-note">Repository is empty — upload circulars to begin.</div>
      ) : (
        <>
          <div className="repo-toolbar">
            <input
              className="repo-search"
              placeholder="Search by filename, circular no., issuer, title, status…"
              value={repoSearch}
              onChange={(e) => setRepoSearch(e.target.value)}
            />
            {isAdmin && selected.size > 0 && (
              <button className="link-danger bulk-delete-btn" onClick={bulkDelete}
                      disabled={bulkDeleting}>
                {bulkDeleting ? 'Deleting…' : `Delete selected (${selected.size})`}
              </button>
            )}
          </div>
          {filteredDocs.length === 0 ? (
            <div className="ok-note">No documents match "{repoSearch}".</div>
          ) : (
          <div className="scroll-window table-window">
            <table>
              <thead>
                <tr>
                  {isAdmin && (
                    <th style={{ width: 28 }}>
                      <input type="checkbox" checked={allVisibleSelected}
                             onChange={toggleAllVisible} title="Select all visible" />
                    </th>
                  )}
                  <th>File</th><th>Circular No.</th><th>Issuer</th>
                  <th>Year</th><th>Status</th><th>Chunks</th><th>Uploaded</th>
                  {isAdmin && <th></th>}
                </tr>
              </thead>
              <tbody>
                {filteredDocs.map((d) => {
                  const [label, cls] = PILL[d.status] || [d.status, 'warn'];
                  return (
                    <tr key={d.id} className={selected.has(d.id) ? 'row-selected' : ''}>
                      {isAdmin && (
                        <td>
                          <input type="checkbox" checked={selected.has(d.id)}
                                 onChange={() => toggleOne(d.id)} />
                        </td>
                      )}
                      <td>
                        {d.stored_name ? (
                          <button className="file-link" title={`Preview circular — originally uploaded as ${d.filename}`}
                                  onClick={() => setPreview({
                                    name: d.filename,
                                    url: `/files/${d.stored_name}` })}>
                            {d.title || d.filename}
                          </button>
                        ) : d.filename}
                      </td>
                      <CircularNoCell doc={d} editable={isAdmin} onSaved={onCircularNoSaved} />
                      <td>{(d.issuer || '—').slice(0, 45)}</td>
                      <td>{d.year || '—'}</td>
                      <td><span className={`status-pill ${cls}`} title={d.status_detail || ''}>
                        {label}</span></td>
                      <td>{d.n_chunks}</td>
                      <td>{d.uploaded_at}</td>
                      {isAdmin && (
                        <td>
                          <button className="link-danger" title="Delete circular"
                                  onClick={() => removeDoc(d)}>delete</button>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          )}
        </>
      )}

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
    </div>
  );
}
