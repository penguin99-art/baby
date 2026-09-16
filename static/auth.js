/* Shared authentication helpers for the baby assistant.
   Loaded before app.js / login.js / accounts.js. */
const AUTH_HEADER = {'X-Requested-With': 'BabyAssistant'};

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {...AUTH_HEADER, ...options.headers},
  });
  let data;
  try { data = await res.json(); }
  catch {
    const error = new Error('invalid_response');
    error.status = res.status;
    throw error;
  }
  if (!res.ok) {
    const error = new Error(data.error || 'request_failed');
    error.status = res.status;
    error.code = data.error;
    throw error;
  }
  return data;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;'}[c]));
}

function getMe() {
  return api('/api/auth/me');
}

function login(username, password) {
  return api('/api/auth/login', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, password})});
}

function logout() {
  return api('/api/auth/logout', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
}

function changePassword(currentPassword, newPassword) {
  return api('/api/auth/password', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({current_password: currentPassword, new_password: newPassword})});
}

function bootstrap(setupToken, username, password) {
  return api('/api/auth/bootstrap', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({setup_token: setupToken, username, password})});
}

/* Resolves to the current user object, or null when logged out. */
const authReady = getMe().then(status => status.user).catch(() => null);

function safeNext() {
  const next = new URLSearchParams(location.search).get('next') || 'index.html';
  return /^[a-zA-Z0-9_.-]+\.html$/.test(next) ? next : 'index.html';
}

function redirectToLogin() {
  location.href = `login.html?next=${encodeURIComponent(safeNext())}`;
}

async function logoutAndRedirect() {
  try { await logout(); } catch {}
  location.href = 'login.html';
}