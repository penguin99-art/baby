const $ = (selector) => document.querySelector(selector);
const next = safeNext();

function show(view) {
  for (const id of ['view-login', 'view-change', 'view-bootstrap']) {
    $(`#${id}`).hidden = id !== view;
  }
}

function showError(message) {
  const target = $('.auth-card section:not([hidden]) .auth-error');
  target.textContent = message;
  target.hidden = false;
}

function clearErrors() {
  document.querySelectorAll('.auth-error').forEach(el => { el.hidden = true; });
}

async function init() {
  let status;
  try { status = await getMe(); }
  catch { show('view-login'); return; }
  if (status.user && !status.user.must_change_password) { location.href = next; return; }
  if (status.user && status.user.must_change_password) { show('view-change'); return; }
  if (status.setup_required && status.bootstrap_available) show('view-bootstrap');
  else show('view-login');
}

$('#login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  clearErrors();
  try {
    const result = await login($('#login-username').value.trim(), $('#login-password').value);
    $('#login-password').value = '';
    if (result.user.must_change_password) show('view-change');
    else location.href = next;
  } catch (error) {
    showError(error.status === 429 ? '尝试次数过多，请稍后再试' : '用户名或密码不正确');
  }
});

$('#change-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  clearErrors();
  const current = $('#change-current').value;
  const fresh = $('#change-new').value;
  if (fresh !== $('#change-confirm').value) { showError('两次输入的新密码不一致'); return; }
  try {
    await changePassword(current, fresh);
    location.href = `login.html?next=${encodeURIComponent(next)}`;
  } catch (error) {
    showError(error.status === 401 ? '当前密码不正确' : '新密码不符合要求（12-128 位）');
  }
});

$('#bootstrap-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  clearErrors();
  try {
    await bootstrap($('#bootstrap-token').value.trim(), $('#bootstrap-username').value.trim(), $('#bootstrap-password').value);
    $('#bootstrap-token').value = '';
    $('#bootstrap-password').value = '';
    $('#login-username').value = $('#bootstrap-username').value;
    show('view-login');
    $('#login-error').textContent = '管理员已创建，请登录';
    $('#login-error').hidden = false;
  } catch (error) {
    showError(error.status === 403 ? 'Setup token 不正确' : '创建失败，请检查用户名和密码要求');
  }
});

init();