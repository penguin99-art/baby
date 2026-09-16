const $ = (selector) => document.querySelector(selector);
const messages = $('#messages');
const query = $('#query');
const welcome = messages.innerHTML;
let revision = 0;
let ready = false;
let busy = false;
let retryRequest = null;
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let recording = false;
let voiceCancelled = false;
let voiceDraft = '';
let speechEnabled = false;
let activeAudioButton = null;
let currentUser = null;

function toast(text) {
  const el = $('#toast');
  el.textContent = text;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3000);
}

function authFailure(error) {
  if (error.status === 401 || error.code === 'password_change_required') { redirectToLogin(); return true; }
  if (error.status === 403) { toast('没有权限执行此操作'); return true; }
  return false;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/^### (.+)$/gm, '<h5>$1</h5>')
    .replace(/^## (.+)$/gm, '<h4>$1</h4>').replace(/^# (.+)$/gm, '<h3>$1</h3>')
    .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>').replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/\n{2,}/g, '</p><p>').replace(/\n/g, '<br>');
  return `<p>${html}</p>`;
}

function stopSpeech() {
  window.speechSynthesis?.cancel();
  if (activeAudioButton) activeAudioButton.textContent = '播放';
  activeAudioButton = null;
}

function speak(text, button = null) {
  if (!('speechSynthesis' in window) || recording) return;
  stopSpeech();
  activeAudioButton = button;
  if (button) button.textContent = '停止';
  const utterance = new SpeechSynthesisUtterance(text.replace(/[#*`\[\]]/g, ''));
  utterance.lang = 'zh-CN';
  utterance.rate = 0.95;
  utterance.onend = utterance.onerror = () => {
    if (button) button.textContent = '播放';
    if (activeAudioButton === button) activeAudioButton = null;
  };
  window.speechSynthesis.speak(utterance);
}

function addMessage(text, type, meta = '') {
  const el = document.createElement('div');
  el.className = `message ${type}`;
  if (type.includes('assistant')) el.innerHTML = renderMarkdown(text);
  else el.textContent = text;
  if (meta) {
    const footer = document.createElement('span');
    footer.className = 'meta';
    footer.textContent = meta;
    if ('speechSynthesis' in window) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'audio-button';
      button.textContent = '播放';
      button.title = '朗读这条回答';
      button.onclick = () => activeAudioButton === button ? stopSpeech() : speak(text, button);
      footer.appendChild(button);
    }
    el.appendChild(footer);
  }
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function renderResponse(data) {
  if (!data || typeof data.answer !== 'string') throw new Error('invalid_response');
  const labels = {assistant_identity:'关于我', medical_boundary:'专业边界', companion_with_knowledge:'知识库依据', companion:'陪伴回应', medical_urgent:'就医提示', mental_health_crisis:'安全支持'};
  const modes = {fixed:'固定能力', llm:'模型生成', fallback:'本地降级'};
  const execution = data.execution || {};
  const knowledge = {deferred:'倾听优先', insufficient_evidence:'依据不足', generation_unavailable:'知识回答暂不可用', output_blocked:'输出已拦截'};
  const meta = [labels[data.route] || '回复', modes[execution.mode] || '历史回复', knowledge[data.knowledge_status]].filter(Boolean).join(' · ');
  return addMessage(data.answer, data.status === 'supported' ? 'assistant supported' : 'assistant refused', meta);
}

function renderCitations(items) {
  $('#citations').innerHTML = items.length ? items.map((c, i) => `<article class="citation"><span class="citation-score">${Number(c.score).toFixed(2)}</span><div class="citation-title">[${escapeHtml(c.citation_number || i + 1)}] ${escapeHtml(c.name)}</div><small>${escapeHtml(c.chunk_id)}</small><blockquote>${escapeHtml(c.text)}</blockquote></article>`).join('') : '<div class="empty-evidence">本次没有使用知识库证据。</div>';
}

function updateControls() {
  const blocked = busy || !ready;
  $('#ask-form .send').disabled = blocked || recording;
  $('#clear-chat').disabled = blocked || recording;
  query.disabled = blocked;
  $('#voice-btn').disabled = blocked || !Recognition;
  document.querySelectorAll('.suggestions button').forEach(button => { button.disabled = blocked || recording; });
}

function applyRole(user) {
  const admin = user.role === 'admin';
  $('#settings-open').hidden = !admin;
  $('#import-panel').hidden = !admin;
  $('#reset-btn').hidden = !admin;
  $('#accounts-link').hidden = !admin;
  $('#user-chip').hidden = false;
  $('#user-name').textContent = user.username;
  $('#user-role').textContent = admin ? '管理员' : '体验账号';
}

function wipeSensitiveDom() {
  messages.innerHTML = '';
  $('#citations').innerHTML = '';
  $('#doc-list').innerHTML = '';
  $('#doc-count').textContent = '0';
  $('#chunk-count').textContent = '0';
  query.value = '';
  retryRequest = null;
  revision = 0;
  ready = false;
  stopSpeech();
}

async function revalidateSession() {
  if (!currentUser) return;
  try {
    const status = await getMe();
    if (!status.user || status.user.id !== currentUser.id) {
      wipeSensitiveDom();
      redirectToLogin();
    }
  } catch {}
}

async function loadSession() {
  const state = await api('/api/session');
  revision = state.revision;
  messages.innerHTML = state.messages.length ? '' : welcome;
  let citations = [];
  for (const item of state.messages) {
    if (item.role === 'user') addMessage(item.content, 'user');
    else if (item.response) {
      renderResponse(item.response);
      citations = item.response.citations || [];
    }
  }
  renderCitations(citations);
  ready = true;
  updateControls();
}

$('#ask-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = query.value.trim();
  if (!text || busy || !ready || recording) return;
  if (text.length > 500) return toast('每次最多发送 500 字，请分开说。');
  stopSpeech();
  if (!retryRequest || retryRequest.query !== text) {
    retryRequest = {query:text, revision, request_id:crypto.randomUUID()};
  }
  busy = true;
  updateControls();
  $('.welcome')?.remove();
  const user = addMessage(text, 'user');
  const pending = addMessage('正在回复…', 'assistant');
  try {
    const data = await api('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(retryRequest)});
    if (typeof data.answer !== 'string') throw new Error('invalid_response');
    pending.remove();
    revision = data.revision;
    retryRequest = null;
    query.value = '';
    renderResponse(data);
    renderCitations(data.citations || []);
    if (speechEnabled) speak(data.answer);
  } catch (error) {
    pending.remove();
    user.remove();
    if (authFailure(error)) return;
    if (error.status === 409) {
      retryRequest = null;
      try { await loadSession(); }
      catch (loadError) { if (authFailure(loadError)) return; ready = false; }
      toast('会话已更新，草稿保留；请确认后重新发送。');
    } else toast('暂时无法完成，草稿已保留，可以重试。');
  } finally {
    busy = false;
    updateControls();
  }
});

messages.addEventListener('click', event => {
  const button = event.target.closest('.suggestions button');
  if (!button || busy || !ready) return;
  query.value = button.textContent.trim();
  $('#ask-form').requestSubmit();
});

$('#clear-chat').onclick = async () => {
  if (busy || recording || !confirm('删除本地保存的这段会话？知识库不会被删除。')) return;
  busy = true; updateControls(); stopSpeech();
  try {
    await api('/api/session/clear', {method:'POST'});
    retryRequest = null;
    query.value = '';
    await loadSession();
    toast('会话已删除');
  } catch (error) { if (authFailure(error)) return; toast('删除失败，请重试'); }
  finally { busy = false; updateControls(); }
};

async function loadDocs() {
  const data = await api('/api/docs');
  $('#doc-count').textContent = data.docs.length;
  $('#chunk-count').textContent = data.docs.reduce((n, doc) => n + doc.chunks, 0);
  $('#doc-list').innerHTML = data.docs.map(doc => `<div class="doc"><div class="doc-name">${escapeHtml(doc.name)}</div><div class="doc-meta">${doc.chunks} 个分块</div></div>`).join('') || '<div class="empty-evidence">还没有文档。</div>';
}

async function upload(files) {
  if (!files.length) return;
  const form = new FormData();
  [...files].forEach(file => form.append('files', file));
  try {
    const data = await api('/api/upload', {method:'POST', body:form});
    toast(data.added?.length ? `已导入 ${data.added.length} 篇文档` : '没有可导入的 Markdown 文件');
    await loadDocs();
  } catch (error) { if (authFailure(error)) return; toast('导入失败，请重试'); }
}
$('#file-input').onchange = event => upload(event.target.files);
const dropzone = $('#dropzone');
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => {
  event.preventDefault();
  dropzone.classList.toggle('drag', name === 'dragenter' || name === 'dragover');
}));
dropzone.addEventListener('drop', event => upload(event.dataTransfer.files));
$('#reset-btn').onclick = async () => {
  if (!confirm('确定清空全部知识库？会话不会被删除。')) return;
  try { await api('/api/reset', {method:'POST'}); await loadDocs(); }
  catch (error) { if (authFailure(error)) return; toast('清空失败，请重试'); }
};

const overlay = $('#settings-overlay');
$('#settings-open').onclick = async () => {
  overlay.classList.add('open'); overlay.setAttribute('aria-hidden', 'false');
  try {
    const config = await api('/api/config');
    for (const kind of ['llm', 'embedding']) {
      $(`#${kind}-base`).value = config[`${kind}_base_url`] || '';
      $(`#${kind}-model`).value = config[`${kind}_model`] || '';
      $(`#${kind}-key`).value = '';
      $(`#${kind}-key`).placeholder = config[`${kind}_api_key`] ? '已配置，留空保留' : '未配置';
    }
  } catch (error) { if (authFailure(error)) return; toast('配置读取失败'); }
};
$('#settings-close').onclick = () => {
  overlay.classList.remove('open'); overlay.setAttribute('aria-hidden', 'true');
  $('#llm-key').value = ''; $('#embedding-key').value = '';
};
$('#settings-form').onsubmit = async event => {
  event.preventDefault();
  const payload = {};
  for (const kind of ['llm', 'embedding']) {
    payload[`${kind.toUpperCase()}_BASE_URL`] = $(`#${kind}-base`).value.trim();
    payload[`${kind.toUpperCase()}_MODEL`] = $(`#${kind}-model`).value.trim();
    if ($(`#${kind}-key`).value) payload[`${kind.toUpperCase()}_API_KEY`] = $(`#${kind}-key`).value;
  }
  try {
    await api('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    $('#llm-key').value = ''; $('#embedding-key').value = '';
    toast('配置已保存，仅本次服务进程生效');
  } catch (error) { if (authFailure(error)) return; toast('配置保存失败，请检查服务地址'); }
};
$('#config-test').onclick = async () => {
  $('#config-status').textContent = '正在测试已保存配置…';
  try {
    const data = await api('/api/config/test', {method:'POST'});
    $('#config-status').textContent = `LLM ${data.llm ? '已连接' : '未连接'} · Embedding ${data.embedding ? '已连接' : '未连接'}`;
  } catch (error) { if (authFailure(error)) return; $('#config-status').textContent = '连接测试失败'; }
};

const voiceBtn = $('#voice-btn');
const speechStatus = $('#speech-status-text');
const speechToggle = $('#speech-toggle');
if (Recognition) {
  recognition = new Recognition();
  recognition.lang = 'zh-CN'; recognition.interimResults = true; recognition.continuous = false;
  recognition.onstart = () => { speechStatus.textContent = '正在听…'; };
  recognition.onresult = event => {
    if (voiceCancelled) return;
    const transcript = Array.from(event.results, result => result[0].transcript).join('');
    query.value = (voiceDraft + (voiceDraft ? ' ' : '') + transcript).slice(0, 500);
  };
  recognition.onerror = event => {
    voiceCancelled = true;
    toast(event.error === 'not-allowed' ? '麦克风权限未开启' : '语音识别未完成，文字已保留');
  };
  recognition.onend = () => {
    recording = false; voiceBtn.classList.remove('listening');
    speechStatus.textContent = '语音输入已就绪';
    voiceBtn.setAttribute('aria-pressed', 'false');
    updateControls(); query.focus();
  };
  voiceBtn.onclick = () => {
    if (recording) { voiceCancelled = true; recognition.abort(); return; }
    if (busy || !ready) return;
    stopSpeech(); voiceDraft = query.value; voiceCancelled = false; recording = true;
    voiceBtn.classList.add('listening'); voiceBtn.setAttribute('aria-pressed', 'true');
    updateControls();
    try { recognition.start(); }
    catch { recording = false; voiceBtn.classList.remove('listening'); updateControls(); toast('语音暂时不可用'); }
  };
  speechStatus.textContent = '语音输入已就绪';
} else speechStatus.textContent = '当前浏览器不支持语音输入';
speechToggle.textContent = '自动朗读：关';
speechToggle.setAttribute('aria-pressed', 'false');
speechToggle.disabled = !('speechSynthesis' in window);
speechToggle.onclick = () => {
  speechEnabled = !speechEnabled;
  speechToggle.textContent = `自动朗读：${speechEnabled ? '开' : '关'}`;
  speechToggle.setAttribute('aria-pressed', String(speechEnabled));
  if (!speechEnabled) stopSpeech();
};
window.addEventListener('pagehide', () => { voiceCancelled = true; recognition?.abort(); stopSpeech(); });
window.addEventListener('pageshow', revalidateSession);
document.addEventListener('visibilitychange', () => { if (!document.hidden) revalidateSession(); });
$('#logout-btn').onclick = async () => {
  wipeSensitiveDom();
  await logoutAndRedirect();
};
updateControls();
authReady.then(user => {
  if (!user || user.must_change_password) { redirectToLogin(); return; }
  currentUser = user;
  applyRole(user);
  loadDocs().catch(error => { if (authFailure(error)) return; toast('知识库读取失败'); });
  loadSession().catch(error => { if (authFailure(error)) return; toast('会话读取失败，请刷新重试'); });
});
