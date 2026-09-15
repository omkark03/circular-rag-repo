import React, { useCallback, useEffect, useState } from 'react';
import { api, setUnauthorizedHandler } from './api.js';
import Header from './components/Header.jsx';
import Sidebar from './components/Sidebar.jsx';
import Chat from './components/Chat.jsx';
import Upload from './components/Upload.jsx';
import Dashboard from './components/Dashboard.jsx';
import Admin from './components/Admin.jsx';
import Templates from './components/Templates.jsx';
import Login from './components/Login.jsx';

export default function App() {
  const [user, setUser] = useState(null);          // {username, role}
  const [authChecked, setAuthChecked] = useState(false);
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [view, setView] = useState('chat');        // chat|upload|dashboard|admin
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  // Sidebar: pinned open on wide screens, auto-hidden (overlay) on narrow
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 900);
  const narrow = () => window.innerWidth <= 900;

  const logout = useCallback(() => {
    localStorage.removeItem('token');
    setUser(null);
    setChats([]);
    setActiveChat(null);
  }, []);

  useEffect(() => { setUnauthorizedHandler(logout); }, [logout]);

  // Restore session
  useEffect(() => {
    if (localStorage.getItem('token')) {
      api.me().then(setUser).catch(() => logout()).finally(() => setAuthChecked(true));
    } else setAuthChecked(true);
  }, [logout]);

  // Auto-hide sidebar when the window becomes narrow
  useEffect(() => {
    const onResize = () => { if (narrow()) setSidebarOpen(false); };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const refreshChats = useCallback(() => api.listChats().then(setChats), []);
  const refreshStats = useCallback(
    () => api.dashboard().then((d) => setStats(d.stats)).catch(() => {}), []);

  useEffect(() => {
    if (!user) return;
    refreshChats();
    refreshStats();
    api.health().then(setHealth).catch(() => setHealth({ ok: false }));
  }, [user, refreshChats, refreshStats]);

  if (!authChecked) return null;
  if (!user) return <Login onLogin={setUser} />;

  const navigate = (v) => { setView(v); if (narrow()) setSidebarOpen(false); };

  const newChat = async () => {
    const c = await api.newChat();
    await refreshChats();
    setActiveChat(c.id);
    navigate('chat');
  };

  const deleteChat = async (id) => {
    await api.deleteChat(id);
    if (id === activeChat) setActiveChat(null);
    refreshChats();
  };

  return (
    <div className="shell">
      <Header
        user={user}
        health={health}
        onToggleSidebar={() => setSidebarOpen((o) => !o)}
        onLogout={logout}
      />
      <div className="body">
        {sidebarOpen && narrow() && (
          <div className="scrim" onClick={() => setSidebarOpen(false)} />
        )}
        <Sidebar
          open={sidebarOpen}
          overlay={narrow()}
          role={user.role}
          chats={chats}
          activeChat={view === 'chat' ? activeChat : null}
          view={view}
          stats={stats}
          onNewChat={newChat}
          onSelectChat={(id) => { setActiveChat(id); navigate('chat'); }}
          onDeleteChat={deleteChat}
          onNavigate={navigate}
          onChatsChanged={refreshChats}
        />
        <div className="main">
          {view === 'chat' && (
            <Chat chatId={activeChat} onFirstMessage={refreshChats} onNeedChat={newChat} />
          )}
          {view === 'upload' && (user.role === 'admin' || user.role === 'uploader')
            && <Upload onDone={refreshStats} />}
          {view === 'dashboard' && <Dashboard role={user.role} onChanged={refreshStats} />}
          {view === 'admin' && user.role === 'admin' && <Admin />}
          {view === 'templates' && <Templates onPreview={() => {}} />}
        </div>
      </div>
    </div>
  );
}
