import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '../api.js';

export default function Templates({ onPreview }) {
  const [mode, setMode] = useState('template'); // 'template' | 'schedule'
  const [presets, setPresets] = useState([]);
  const [docType, setDocType] = useState('');
  const [custom, setCustom] = useState('');
  const [details, setDetails] = useState('');
  const [stationsText, setStationsText] = useState('');
  const [startDate, setStartDate] = useState('');
  const [content, setContent] = useState('');
  const [refs, setRefs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState('');
  const [overrideRegular, setOverrideRegular] = useState('');
  const [overrideImportant, setOverrideImportant] = useState('');

  useEffect(() => {
    api.templatePresets().then((p) => {
      setPresets(p);
      if (p.length) setDocType(p[0].key);
    }).catch(() => {});
  }, []);

  const effectiveLabel = () => {
    if (mode === 'schedule') return 'Inspection Schedule';
    if (docType === '__custom__') return custom.trim();
    return presets.find((p) => p.key === docType)?.label || docType;
  };

  const stationsList = () =>
    stationsText.split('\n').map((s) => s.trim()).filter(Boolean);

  const canGenerate = () =>
    !busy && (mode === 'schedule'
      ? stationsList().length > 0 && !!startDate
      : !!effectiveLabel());

  const scheduleToMarkdown = (result) => {
    const row = (r) =>
      `| ${r.station} | ${r.important ? 'Important' : 'Regular'} | ${r.start_date} | ${r.end_date} | ${r.duration_days} day(s) |`;
    const header = '| Station | Category | Start Date | End Date | Duration |\n|---|---|---|---|---|';
    if (!result.ok) {
      let md = `**⚠️ ${result.error}**\n`;
      if (result.partial_schedule && result.partial_schedule.length) {
        md += `\n${header}\n` + result.partial_schedule.map(row).join('\n');
        md += `\n\n**Stations that do NOT fit in the window:** ${result.stations_that_dont_fit.join(', ')}`;
      }
      if (result.rules_found) {
        const rf = result.rules_found;
        md += `\n\nRules found so far — regular: ${rf.regular_duration_days ?? 'not found'}, ` +
              `important: ${rf.important_duration_days ?? 'not found'}.` +
              (rf.source_note ? ` (${rf.source_note})` : '') +
              '\n\nProvide the missing duration(s) manually below and try again.';
      }
      return md;
    }
    let md = `Regular station duration: **${result.regular_duration_days} day(s)** · ` +
             `Important station duration: **${result.important_duration_days} day(s)**`;
    if (result.rules_source) md += ` — *${result.rules_source}*`;
    md += `\n\n${header}\n` + result.schedule.map(row).join('\n');
    return md;
  };

  const generate = async () => {
    if (!canGenerate()) return;
    setBusy(true);
    setError('');
    setContent('');
    setRefs([]);
    try {
      if (mode === 'schedule') {
        const overrides = {};
        if (overrideRegular) overrides.regular_duration_days = parseInt(overrideRegular, 10);
        if (overrideImportant) overrides.important_duration_days = parseInt(overrideImportant, 10);
        const result = await api.generateScheduleDates(stationsList(), startDate, details, overrides);
        setContent(scheduleToMarkdown(result));
        setRefs(result.references || []);
      } else {
        await api.generateTemplateStream(docType === '__custom__' ? custom.trim() : docType, details, {
          onRefs: setRefs,
          onToken: (t) => setContent((c) => c + t),
        });
      }
    } catch (e) {
      setError(`⚠️ ${e.message}. Is the backend running?`);
    } finally {
      setBusy(false);
    }
  };

  const exportDocx = async () => {
    if (!content.trim() || exporting) return;
    setExporting(true);
    try {
      await api.exportTemplate(effectiveLabel(), content);
    } catch (e) {
      setError(`⚠️ Export failed: ${e.message}`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="main-scroll">
      <h1>Templates</h1>
      <p style={{ color: 'var(--muted)' }}>
        Generate a fillable document template, or an actual inspection
        schedule assigning your real police stations to months — both
        grounded in whatever your circulars actually prescribe.
      </p>

      <div className="tpl-mode-toggle">
        <button className={mode === 'template' ? 'active' : ''}
                onClick={() => setMode('template')}>📄 Document template</button>
        <button className={mode === 'schedule' ? 'active' : ''}
                onClick={() => setMode('schedule')}>🗓️ Inspection schedule</button>
      </div>

      {mode === 'template' ? (
        <div className="tpl-controls">
          <select value={docType} onChange={(e) => setDocType(e.target.value)}>
            {presets.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
            <option value="__custom__">Custom…</option>
          </select>
          {docType === '__custom__' && (
            <input placeholder="e.g. Seizure Memo, Site Visit Report"
                   value={custom} onChange={(e) => setCustom(e.target.value)} />
          )}
          <input
            className="tpl-details"
            placeholder="Optional context, e.g. 'for a warehouse inspection under the Weights & Measures Act'"
            value={details} onChange={(e) => setDetails(e.target.value)}
          />
          <button className="upload-btn" style={{ marginTop: 0 }}
                  onClick={generate} disabled={!canGenerate()}>
            {busy ? 'Drafting…' : 'Generate template'}
          </button>
        </div>
      ) : (
        <div className="tpl-controls tpl-schedule-controls">
          <p style={{ color: 'var(--muted)', marginTop: 0, marginBottom: 8 }}>
            List your real police stations below, one per line. Mark any that
            need extra inspection time with <code>[important]</code> — e.g.
            <code> Andheri Police Station [important]</code>. The schedule
            will use only these stations; none will be invented, and none
            will be silently dropped. Dates are computed exactly (not
            guessed by the AI) from the actual duration/frequency rules
            found in your circulars.
          </p>
          <textarea
            className="tpl-stations"
            rows={6}
            placeholder={'Andheri Police Station [important]\nBandra Police Station\nColaba Police Station\nDadar Police Station'}
            value={stationsText}
            onChange={(e) => setStationsText(e.target.value)}
          />
          <label className="tpl-date-label">
            Start date
            <input type="date" className="tpl-date" value={startDate}
                   onChange={(e) => setStartDate(e.target.value)} />
          </label>
          <input
            className="tpl-details"
            placeholder="Optional context, e.g. 'District: Mumbai'"
            value={details} onChange={(e) => setDetails(e.target.value)}
          />
          <details className="tpl-overrides">
            <summary>Manually set durations (only if your circulars don't specify one)</summary>
            <div className="tpl-override-row">
              <label>Regular station: <input type="number" min="1" placeholder="days"
                     value={overrideRegular}
                     onChange={(e) => setOverrideRegular(e.target.value)} /></label>
              <label>Important station: <input type="number" min="1" placeholder="days"
                     value={overrideImportant}
                     onChange={(e) => setOverrideImportant(e.target.value)} /></label>
            </div>
          </details>
          <button className="upload-btn" style={{ marginTop: 0 }}
                  onClick={generate} disabled={!canGenerate()}>
            {busy ? 'Computing schedule…' : 'Generate schedule'}
          </button>
        </div>
      )}

      {error && <div className="flag bad" style={{ marginTop: 12 }}>{error}</div>}

      {(content || busy) && (
        <>
          <div className="tpl-preview-head">
            <span className="who">Preview</span>
            <button className="upload-btn" style={{ marginTop: 0 }}
                    onClick={exportDocx} disabled={!content.trim() || exporting}>
              {exporting ? 'Preparing…' : '⬇ Download as Word (.docx)'}
            </button>
          </div>
          <div className="tpl-preview bubble">
            {content ? (
              <div className="md">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ node, ...props }) => (
                      <div className="md-table-wrap"><table {...props} /></div>
                    ),
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            ) : <span className="thinking">
                  {mode === 'schedule' ? 'Reading circulars and computing dates…'
                                       : 'Searching repository and drafting…'}
                </span>}
          </div>

          {refs.length > 0 && (
            <div className="ref-list">
              <div className="who" style={{ marginTop: 12 }}>
                Grounded in these circulars (APA)
              </div>
              {refs.map((r, i) => (
                <div key={i} className="ref-card" role="button" tabIndex={0}
                     onClick={() => onPreview && onPreview({ name: r.title, url: r.url })}>
                  {r.circular_no && <div className="circ-no">CIRCULAR NO. {r.circular_no}</div>}
                  {r.issuer}. ({r.year}). <i>{r.title}</i>.
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
