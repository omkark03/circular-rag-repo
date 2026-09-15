import React, { useState } from 'react';
import { api } from '../api.js';
import { IconArchive, IconDocument, IconUpload, IconSettings, IconChatPlus } from './icons.jsx';

function ChatRow({ c, active, onSelect, onDelete, onRenamed }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(c.title);

  const commit = async () => {
    const title = value.trim();
    setEditing(false);
    if (!title || title === c.title) { setValue(c.title); return; }
    await api.renameChat(c.id, title);
    onRenamed();
  };

  if (editing) {
    return (
      <div className="chat-item editing">
        <input
          className="chat-rename-input"
          value={value}
          autoFocus
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit();
            if (e.key === 'Escape') { setValue(c.title); setEditing(false); }
          }}
          onBlur={commit}
        />
      </div>
    );
  }

  return (
    <div
      className={`chat-item ${active ? 'active' : ''}`}
      onClick={() => onSelect(c.id)}
      onDoubleClick={(e) => { e.stopPropagation(); setEditing(true); }}
    >
      <span title={`${c.title} (double-click to rename)`}>{c.title}</span>
      <button
        className="del" title="Delete conversation"
        onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
      >✕</button>
    </div>
  );
}

export default function Sidebar({
  open, overlay, role, chats, activeChat, view, stats,
  onNewChat, onSelectChat, onDeleteChat, onNavigate, onChatsChanged,
}) {
  const failed = stats?.failed || 0;
  const missing = stats?.missing_refs || 0;
  const canUpload = role === 'admin' || role === 'uploader';

  return (
    <aside className={`sidebar ${open ? 'open' : 'closed'} ${overlay ? 'overlay' : ''}`}>
      <div className="side-nav side-nav-top">
        <div className={`chat-item badge-row ${view === 'dashboard' ? 'active' : ''}`}
             onClick={() => onNavigate('dashboard')}>
          <span><IconArchive className="nav-icon" /> Circulars</span>
          {failed + missing > 0 && <b>{failed + missing}</b>}
        </div>
        
        {canUpload && (
          <div className={`chat-item ${view === 'upload' ? 'active' : ''}`}
               onClick={() => onNavigate('upload')}>
            <span><IconUpload className="nav-icon" /> Upload circulars</span>
          </div>
        )}
        {role === 'admin' && (
          <div className={`chat-item ${view === 'admin' ? 'active' : ''}`}
               onClick={() => onNavigate('admin')}>
            <span><IconSettings className="nav-icon" /> Administration</span>
          </div>
        )}
      </div>

      <button className="new-chat" onClick={onNewChat}>
        <IconChatPlus className="nav-icon" /> New conversation
      </button>

      <div className="side-label">Conversations</div>
      <div className="chat-list">
        {chats.length === 0 && (
          <div className="chat-item" style={{ color: '#7e948a', cursor: 'default' }}>
            <span>No conversations yet</span>
          </div>
        )}
        {chats.map((c) => (
          <ChatRow
            key={c.id}
            c={c}
            active={c.id === activeChat}
            onSelect={onSelectChat}
            onDelete={onDeleteChat}
            onRenamed={onChatsChanged}
          />
        ))}
      </div>
    </aside>
  );
}
