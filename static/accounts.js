const $ = (selector) => document.querySelector(selector);
const usersBody = $('#users-body');

function showError(message) {
  const el = $('#create-error');
  el.textContent = message;
  el.hidden = false;
}

async function loadUsers() {
  const data = await api('/api/admin/users');
  usersBody.innerHTML = data.users.map(user => {
    const actions = user.role === 'admin'
      ? '<span class="muted">—</span>'
      : `<button class="text-button" data-action="toggle" data-id="${user.id}" data-active="${user.active}">${user.active ? '禁用' : '启用'}</button>
         <button class="text-button" data-action="reset" data-id="${user.id}">重置密码</button>`;
    return `<tr>
      <td>${escapeHtml(user.username)}</td>
      <td>${user.role === 'admin' ? '管理员' : '体验账号'}</td>
      <td>${user.active ? '正常' : '已禁用'}</td>
      <td>${escapeHtml(user.created_at)}</td>
      <td class="row-actions">${actions}</td>
    </tr>`;
  }).join('');
}

async function init() {
  let status;
  try { status = await getMe(); }
  catch { redirectToLogin(); return; }
  if (!status.user) { redirectToLogin(); return; }
  if (status.user.role !== 'admin') { location.href = 'index.html'; return; }
  try { await loadUsers(); }
  catch { showError('账号列表读取失败'); }
}

$('#create-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const el = $('#create-error');
  el.hidden = true;
  try {
    await api('/api/admin/users', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username: $('#new-username').value.trim(), password: $('#new-password').value})});
    $('#new-username').value = '';
    $('#new-password').value = '';
    await loadUsers();
  } catch (error) {
    showError(error.status === 409 ? '用户名已存在' : '创建失败，请检查用户名和密码要求');
  }
});

usersBody.addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const id = button.dataset.id;
  try {
    if (button.dataset.action === 'toggle') {
      await api('/api/admin/users/status', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({user_id: id, active: button.dataset.active !== 'true'})});
    } else {
      const password = prompt('输入新的初始密码（12-128 位）。该账号的所有登录会被退出。');
      if (!password) return;
      await api('/api/admin/users/password', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({user_id: id, password})});
    }
    await loadUsers();
  } catch (error) {
    showError(error.status === 403 ? '没有权限' : '操作失败，请重试');
  }
});

$('#logout-btn').addEventListener('click', logoutAndRedirect);

init();