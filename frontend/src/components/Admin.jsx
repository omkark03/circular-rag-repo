import React, { useCallback, useEffect, useState } from 'react';
import { api } from '../api.js';

export default function Admin() {
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({ username: '', password: '', role: 'viewer' });
  const [msg, setMsg] = useState('');
  const [reindexing, setReindexing] = useState(false);
  const [backingUp, setBackingUp] = useState(false);
  const [newPw, setNewPw] = useState('');
  const [xrefEnabled, setXrefEnabled] = useState(null);
  const [xrefSaving, setXrefSaving] = useState(false);

  const load = useCallback(() => api.users().then(setUsers).catch(() => {}), []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    api.getCrossReferenceSetting().then((r) => setXrefEnabled(r.enabled)).catch(() => {});
  }, []);

  const toggleXref = async () => {
    setXrefSaving(true);
    try {
      const r = await api.setCrossReferenceSetting(!xrefEnabled);
      setXrefEnabled(r.enabled);
    } catch (e) {
      setMsg(`⚠️ ${e.message}`);
    } finally {
      setXrefSaving(false);
    }
  };

  const addUser = async () => {
    setMsg('');
    try {
      await api.addUser(form);
      setForm({ username: '', password: '', role: 'viewer' });
      setMsg('✅ User created');
      load();
    } catch (e) { setMsg(`⚠️ ${e.message}`); }
  };

  const removeUser = async (id) => {
    try { await api.deleteUser(id); load(); }
    catch (e) { setMsg(`⚠️ ${e.message}`); }
  };

  const changePw = async () => {
    setMsg('');
    try {
      await api.changePassword(newPw);
      setNewPw('');
      setMsg('✅ Your password was changed');
    } catch (e) { setMsg(`⚠️ ${e.message}`); }
  };

  const [rescanning, setRescanning] = useState(false);
  const [testText, setTestText] = useState('');
  const [testResult, setTestResult] = useState(null);

  const runTest = async () => {
    if (!testText.trim()) return;
    try { setTestResult(await api.testDetection(testText)); }
    catch (e) { setMsg(`⚠️ ${e.message}`); }
  };
  const rescan = async () => {
    setRescanning(true);
    setMsg('');
    try {
      const r = await api.rescanReferences();
      setMsg(`✅ Re-scanned ${r.rescanned} document(s) · resolved ${r.resolved} reference(s)`
             + (r.failed ? ` · ${r.failed} failed` : ''));
    } catch (e) { setMsg(`⚠️ ${e.message}`); }
    finally { setRescanning(false); }
  };

  const downloadBackup = async () => {
    setBackingUp(true);
    setMsg('');
    try {
      await api.downloadBackup();
    } catch (e) {
      setMsg(`⚠️ Backup failed: ${e.message}`);
    } finally {
      setBackingUp(false);
    }
  };

  const reindex = async () => {
    setReindexing(true);
    setMsg('');
    try {
      const r = await api.reindex();
      setMsg(`✅ Re-embedded ${r.reindexed} document(s) with ${r.embed_model}`
             + (r.failed ? ` · ${r.failed} failed (see dashboard)` : ''));
    } catch (e) { setMsg(`⚠️ ${e.message}`); }
    finally { setReindexing(false); }
  };

  return (
    <div className="main-scroll">
      <h1>Admin</h1>
      {msg && <div className="ok-note" style={{ marginBottom: 14 }}>{msg}</div>}

      <h3>👥 Users & roles</h3>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        admin: everything · uploader: upload + chat · viewer: chat and dashboard only.
      </p>
      <table>
        <thead><tr><th>User</th><th>Role</th><th>Created</th><th></th></tr></thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.id}>
              <td>{u.username}</td>
              <td><span className="status-pill ok">{u.role}</span></td>
              <td>{u.created_at}</td>
              <td><button className="link-danger" onClick={() => removeUser(u.id)}>remove</button></td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="form-row">
        <input placeholder="Username" value={form.username}
               onChange={(e) => setForm({ ...form, username: e.target.value })} />
        <input placeholder="Password (min 6)" type="password" value={form.password}
               onChange={(e) => setForm({ ...form, password: e.target.value })} />
        <select value={form.role}
                onChange={(e) => setForm({ ...form, role: e.target.value })}>
          <option value="viewer">viewer</option>
          <option value="uploader">uploader</option>
          <option value="admin">admin</option>
        </select>
        <button className="upload-btn" style={{ marginTop: 0 }} onClick={addUser}>
          Add user
        </button>
      </div>

      <h3 className="section-title">🔐 Change my password</h3>
      <div className="form-row">
        <input placeholder="New password (min 6)" type="password" value={newPw}
               onChange={(e) => setNewPw(e.target.value)} />
        <button className="upload-btn" style={{ marginTop: 0 }} onClick={changePw}>
          Change
        </button>
      </div>

      <h3 className="section-title">🔗 Cross-reference checking</h3>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        Matching citation formats across decades of inconsistent historical
        documents doesn't scale — but it's reliable for newly authored
        circulars that include an explicit <b>Reference</b> / <b>संदर्भ</b> line.
        When enabled, the dashboard shows missing-reference flags and new
        uploads are scanned for citations. When disabled, that dashboard
        section is hidden and new uploads skip citation extraction entirely
        (a document's own circular number is still always detected, since
        that's needed for chat citations regardless of this setting).
      </p>
      {xrefEnabled === null ? (
        <p style={{ color: 'var(--muted)' }}>Loading…</p>
      ) : (
        <label className="toggle-row">
          <input type="checkbox" checked={xrefEnabled}
                 onChange={toggleXref} disabled={xrefSaving} />
          <span>
            Cross-reference checking is <b>{xrefEnabled ? 'enabled' : 'disabled'}</b>
            {xrefSaving && ' — saving…'}
          </span>
        </label>
      )}

      <h3 className="section-title">💾 Repository backup</h3>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        Downloads a .zip containing the database (users, chats, documents,
        circular numbers, references, settings) plus every stored circular
        file — everything needed to restore this repository elsewhere. The
        search index isn't included (it's large and model-specific); after
        restoring, run "Rebuild search index" to recreate it.
      </p>
      <button className="upload-btn" onClick={downloadBackup} disabled={backingUp}>
        {backingUp ? 'Preparing backup…' : '⬇ Download backup'}
      </button>

      <h3 className="section-title">🧠 Repository index</h3>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        Re-embeds every stored document with the current embedding model.
        Run this after changing <code>EMBED_MODEL</code> (e.g. switching to the
        multilingual embedder for Marathi) — no need to re-upload files.
        This is how repository knowledge is “trained” into the search index;
        the LLM itself stays frozen, which prevents it inventing circular numbers.
      </p>
      <button className="upload-btn" onClick={reindex} disabled={reindexing}>
        {reindexing ? 'Re-embedding repository…' : 'Rebuild search index'}
      </button>

      <h3 className="section-title">🔗 Re-scan cross-references</h3>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        Re-runs circular-number detection and citation extraction on all stored
        documents with the latest patterns, then re-resolves missing references.
        Run this after a detector update instead of re-uploading files.
      </p>
      <button className="upload-btn" onClick={rescan} disabled={rescanning}>
        {rescanning ? 'Re-scanning references…' : 'Re-scan references'}
      </button>

      <h3 className="section-title">🧪 Test number detection</h3>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        Paste text from a circular (Marathi or English) to see exactly which
        circular numbers the detector finds. If a number is missed, copy the
        surrounding line here to verify — and report the exact text so
        patterns can be extended.
      </p>
      <textarea className="test-box" rows={5} value={testText}
                placeholder="e.g. संदर्भ: क्र.पोमसं/१४/६६ अन्वये…"
                onChange={(e) => setTestText(e.target.value)} />
      <button className="upload-btn" onClick={runTest}>Detect numbers</button>
      {testResult && (
        <div className="ok-note" style={{ marginTop: 10 }}>
          <b>Own number:</b> {testResult.own_number || '— none detected —'}<br />
          <b>Referenced numbers:</b>{' '}
          {testResult.references.length === 0 ? '— none —'
            : testResult.references.map((x) => x.normalized).join(' · ')}
        </div>
      )}
    </div>
  );
}
