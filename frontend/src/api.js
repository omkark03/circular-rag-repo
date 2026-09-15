// Fetch wrapper with auth token. Token kept in localStorage (this is a
// locally-run Vite app, not a sandboxed artifact).
let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn; };

const headers = (extra = {}) => {
  const t = localStorage.getItem('token');
  return { ...(t ? { Authorization: `Bearer ${t}` } : {}), ...extra };
};

const j = async (r) => {
  if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
  if (!r.ok) {
    let msg = `API ${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
};

export const api = {
  login: (username, password) =>
    fetch('/api/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }).then(j),
  me: () => fetch('/api/auth/me', { headers: headers() }).then(j),
  changePassword: (password) =>
    fetch('/api/auth/password', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ password }),
    }).then(j),

  health: () => fetch('/api/health').then(j),

  listChats: () => fetch('/api/chats', { headers: headers() }).then(j),
  newChat: () => fetch('/api/chats', { method: 'POST', headers: headers() }).then(j),
  renameChat: (id, title) =>
    fetch(`/api/chats/${id}`, {
      method: 'PATCH', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ title }),
    }).then(j),
  getChat: (id) => fetch(`/api/chats/${id}`, { headers: headers() }).then(j),
  deleteChat: (id) =>
    fetch(`/api/chats/${id}`, { method: 'DELETE', headers: headers() }).then(j),
  // Streaming ask: onRefs(refs) once, onToken(text) per piece, resolves on done
  askStream: async (id, question, { onRefs, onToken, onHits, onTranslating, onTranslation, onTranslationUnavailable, onCheckingContradictions, onContradictions, onExtractingSchedule, onCheckingDurationRules }) => {
    const r = await fetch(`/api/chats/${id}/ask`, {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ question }),
    });
    if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
    if (!r.ok) throw new Error(`API ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'refs') { onRefs(ev.references); onHits && onHits(ev.hits || []); }
        else if (ev.type === 'token') onToken(ev.t);
        else if (ev.type === 'translating') onTranslating && onTranslating();
        else if (ev.type === 'translation') onTranslation && onTranslation(ev.text);
        else if (ev.type === 'translation_unavailable')
          onTranslationUnavailable && onTranslationUnavailable(ev.detail);
        else if (ev.type === 'checking_contradictions')
          onCheckingContradictions && onCheckingContradictions();
        else if (ev.type === 'extracting_schedule')
          onExtractingSchedule && onExtractingSchedule();
        else if (ev.type === 'checking_duration_rules')
          onCheckingDurationRules && onCheckingDurationRules();
        else if (ev.type === 'contradictions')
          onContradictions && onContradictions(ev.items);
      }
    }
  },

  dashboard: () => fetch('/api/dashboard', { headers: headers() }).then(j),
  recheck: () =>
    fetch('/api/dashboard/recheck', { method: 'POST', headers: headers() }).then(j),
  upload: (files) => {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f));
    return fetch('/api/upload', { method: 'POST', headers: headers(), body: fd }).then(j);
  },

  setCircularNo: (id, circularNo) =>
    fetch(`/api/documents/${id}/circular-no`, {
      method: 'PATCH', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ circular_no: circularNo }),
    }).then(j),
  deleteDocument: (id) =>
    fetch(`/api/documents/${id}`, { method: 'DELETE', headers: headers() }).then(j),
  bulkDeleteDocuments: (ids) =>
    fetch('/api/documents/bulk-delete', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ ids }),
    }).then(j),

  users: () => fetch('/api/admin/users', { headers: headers() }).then(j),
  addUser: (u) =>
    fetch('/api/admin/users', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(u),
    }).then(j),
  deleteUser: (id) =>
    fetch(`/api/admin/users/${id}`, { method: 'DELETE', headers: headers() }).then(j),
  templatePresets: () => fetch('/api/templates/presets', { headers: headers() }).then(j),
  generateTemplateStream: async (docType, details, { onRefs, onToken }) => {
    const r = await fetch('/api/templates/generate', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ doc_type: docType, details }),
    });
    if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
    if (!r.ok) throw new Error(`API ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'refs') onRefs(ev.references);
        else if (ev.type === 'token') onToken(ev.t);
      }
    }
  },
  generateScheduleStream: async (stations, details, { onRefs, onToken }) => {
    const r = await fetch('/api/templates/generate-schedule', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ stations, details }),
    });
    if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
    if (!r.ok) throw new Error(`API ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'refs') onRefs(ev.references);
        else if (ev.type === 'token') onToken(ev.t);
      }
    }
  },
  generateScheduleDates: async (stations, startDate, details, overrides = {}) => {
    const r = await fetch('/api/templates/generate-schedule-dates', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ stations, start_date: startDate, details, ...overrides }),
    });
    if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
    if (!r.ok) throw new Error(`API ${r.status}`);
    return r.json();
  },
  exportTemplate: async (title, markdown) => {
    const r = await fetch('/api/templates/export', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ title, markdown }),
    });
    if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
    if (!r.ok) throw new Error(`API ${r.status}`);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title || 'template'}.docx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },
  testDetection: (text) =>
    fetch('/api/admin/test-detection', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ text }),
    }).then(j),
  getCrossReferenceSetting: () =>
    fetch('/api/admin/settings/cross-reference', { headers: headers() }).then(j),
  setCrossReferenceSetting: (enabled) =>
    fetch('/api/admin/settings/cross-reference', {
      method: 'POST', headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ enabled }),
    }).then(j),
  rescanReferences: () =>
    fetch('/api/admin/rescan-references', { method: 'POST', headers: headers() }).then(j),
  downloadBackup: async () => {
    const r = await fetch('/api/admin/backup', { headers: headers() });
    if (r.status === 401) { onUnauthorized(); throw new Error('Session expired'); }
    if (!r.ok) throw new Error(`API ${r.status}`);
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const match = cd.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : 'circular-repository-backup.zip';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },
  reindex: () =>
    fetch('/api/admin/reindex', { method: 'POST', headers: headers() }).then(j),
};
