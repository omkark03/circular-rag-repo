import React, { useState } from 'react';

const ROLE_BADGE = { admin: 'ADMIN', uploader: 'UPLOADER', viewer: 'VIEWER' };

export default function Header({ user, health, onToggleSidebar, onLogout }) {
  const [logoFailed, setLogoFailed] = useState(false);

  return (
    <header className="topbar">
      <button className="hamburger" onClick={onToggleSidebar} title="Toggle sidebar">
        ☰
      </button>
      <div className="topbar-title">
        {logoFailed ? (
          <span className="mark">📜</span>
        ) : (
          <img src="/logo.svg" alt="" className="mark-img"
               onError={() => setLogoFailed(true)} />
        )}
        <h1>Circular Portal</h1>
        <span className="subtitle">XXXXXX</span>
      </div>
      <div className="topbar-right">
        <span className="user-chip">
          {user.username} <b>{ROLE_BADGE[user.role]}</b>
        </span>
        <button className="logout" onClick={onLogout}>Logout</button>
      </div>
    </header>
  );
}
