(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const api = async (url, options = {}) => {
    const response = await fetch(url, options);
    const contentType = response.headers.get('content-type') || '';
    let body;
    if (contentType.includes('application/json')) body = await response.json();
    else body = await response.text();
    if (!response.ok) {
      const detail = body && typeof body === 'object' ? (body.detail || body.error || JSON.stringify(body)) : body;
      throw new Error(detail || `Request failed (${response.status})`);
    }
    return body;
  };
  const postJSON = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const storage = {
    get(key, fallback = null) { try { const v = localStorage.getItem(`aletheia.${key}`); return v === null ? fallback : v; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(`aletheia.${key}`, String(value)); } catch {} }
  };
  const state = {
    currentChatId: storage.get('chatId'), currentChatTitle: 'First conversation', activeView: 'chat', status: null,
    sending: false, listening: false, recognition: null, voiceFinalText: '', micStream: null, audioContext: null, analyser: null, animFrame: 0,
    cameraStream: null, cameraTimer: 0, cameraBusy: false, autoTrain: storage.get('autoTrain', 'true') !== 'false', jobs: [], jobsTimer: 0, statusTimer: 0,
    speechVoices: [], voiceEnabled: storage.get('speakReplies', 'true') === 'true', voiceBackend: storage.get('voiceBackend', 'browser'),
    voiceRate: Number(storage.get('voiceRate', '1')), voicePitch: Number(storage.get('voicePitch', '1')),
    replyTokens: Number(storage.get('replyTokens', '128')), autoScroll: storage.get('autoScroll', 'true') !== 'false',
    cameraEnabled: false, activity: 'idle', currentUserStream: null, lastVoiceText: '',
    currentAudio: null, currentAudioUrl: null, speechResolve: null,
  };
  const supportedFiles = ['.txt', '.md', '.markdown', '.pdf', '.docx', '.epub', '.html', '.htm', '.xhtml', '.parquet'];
  const welcomeTemplate = $('welcomeState')?.cloneNode(true) || null;

  function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function shortNumber(value) {
    if (value === null || value === undefined || value === '') return '—';
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (number >= 1_000_000_000) return `${(number/1_000_000_000).toFixed(1)}B`;
    if (number >= 1_000_000) return `${(number/1_000_000).toFixed(1)}M`;
    if (number >= 10_000) return `${(number/1_000).toFixed(1)}k`;
    return number.toLocaleString();
  }
  function dateLabel(epoch) {
    const date = new Date(Number(epoch) * 1000);
    if (!Number.isFinite(date.getTime())) return '';
    return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function timeLabel(epoch) {
    const date = new Date(Number(epoch) * 1000);
    return Number.isFinite(date.getTime()) ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  }
  function toast(message, kind = 'info', duration = 3900) {
    const region = $('toastRegion');
    const item = document.createElement('div');
    item.className = `toast ${kind}`;
    item.textContent = String(message);
    region.appendChild(item);
    window.setTimeout(() => { item.style.opacity = '0'; item.style.transform = 'translateY(4px)'; window.setTimeout(() => item.remove(), 240); }, duration);
  }
  function setOrbState(activity, customTitle, customDescription) {
    state.activity = activity;
    const orb = $('orbStage');
    orb.dataset.state = activity;
    const labels = {
      idle: ['Quietly present', 'Ready when you are. Start a chat or enable a sensory input.'],
      listening: ['Listening to you', 'Microphone is active. Your browser permission is in control.'],
      thinking: ['Thinking it through', 'Aletheia is processing your message with her current learned state.'],
      speaking: ['Speaking', 'Audio output is active. Click Stop speaking to interrupt it.'],
      learning: ['Learning from material', 'A guarded learning job is updating and evaluating Aletheia.'],
      seeing: ['Observing a scene', 'A camera frame is being encoded and the raw frame will be discarded.'],
      loading: ['Waking up the engine', 'The local model and memory stores are initializing.']
    };
    const config = labels[activity] || labels.idle;
    $('orbStateLabel').textContent = customTitle || config[0];
    $('orbStateDescription').textContent = customDescription || config[1];
    $('stateEqualizer').classList.toggle('active', ['listening','speaking'].includes(activity));
  }
  function setEngineStatus(text, kind = 'normal') {
    $('engineStatusText').textContent = text;
    $('engineStatus').classList.toggle('loading', kind === 'loading');
    $('engineStatus').classList.toggle('error', kind === 'error');
  }
  async function refreshStatus({quiet = true} = {}) {
    try {
      const status = await api('/api/status');
      state.status = status;
      if (status.busy) {
        setEngineStatus('Engine busy', 'loading');
      } else if (status.engine_loaded) {
        setEngineStatus(status.language_ready ? 'Engine ready' : 'Engine learning');
      } else {
        setEngineStatus('Engine not loaded');
      }
      $('metricParams').textContent = shortNumber(status.parameter_count);
      $('metricSteps').textContent = shortNumber(status.training_steps);
      $('metricMemories').textContent = shortNumber(status.memory_count);
      $('metricReady').textContent = status.language_ready ? 'Verified' : 'In progress';
      $('metricReady').style.color = status.language_ready ? 'var(--green)' : 'var(--pink)';
      $('deviceLabel').textContent = status.engine_loaded ? `${status.device || 'CPU'} · ${shortNumber(status.training_tokens)} training tokens` : 'Engine initializes on first interaction';
      $('footerEngineState').textContent = status.busy ? 'Model operation in progress' : status.engine_loaded ? `${status.device || 'Local'} · model loaded` : 'Waiting for first interaction';
      $('replyModeLabel').textContent = status.language_ready ? 'Language readiness verified' : 'Experimental mode';
      if (status.engine_loaded) {
        $('learnedTtsStatus').textContent = status.tts_ready ? `Trained checkpoint available · ${shortNumber(status.tts_steps)} steps` : 'Not ready yet · paired audio/text training required';
        $('learnedTtsBadge').textContent = status.tts_ready ? 'READY' : 'NOT READY';
        $('learnedTtsBadge').classList.toggle('ready', !!status.tts_ready);
        const learnedOption = $('voiceBackendSelect').querySelector('option[value="learned"]');
        learnedOption.disabled = !status.tts_ready;
        if (!status.tts_ready && state.voiceBackend === 'learned') {
          state.voiceBackend = 'browser'; $('voiceBackendSelect').value = 'browser';
          storage.set('voiceBackend', 'browser');
        }
      } else {
        $('learnedTtsStatus').textContent = 'Engine loads on first interaction.';
        $('learnedTtsBadge').textContent = 'UNLOADED';
        $('voiceBackendSelect').querySelector('option[value="learned"]').disabled = true;
      }
      $('chatCountHint').textContent = `${shortNumber(status.conversation_count)} saved conversations`;
      $('memoryChatsMetric').textContent = shortNumber(status.conversation_count);
      $('memoryTotalMetric').textContent = shortNumber(status.memory_count);
      $('memoryLessonsMetric').textContent = shortNumber(status.lessons);
      if (status.worker && status.worker.running && status.worker.current_job_id) {
        if (state.activity !== 'thinking' && state.activity !== 'speaking' && state.activity !== 'listening') setOrbState('learning');
      } else if (state.activity === 'learning') setOrbState('idle');
      if (status.agent_load_error) setEngineStatus('Engine error', 'error');
      if (!quiet && status.engine_loaded) toast('Status refreshed', 'success', 1500);
      if (status.engine_loaded) refreshVoiceStatus();
      return status;
    } catch (error) {
      setEngineStatus('Backend unavailable', 'error');
      $('footerEngineState').textContent = 'Cannot reach local API';
      if (!quiet) toast(`Backend status failed: ${error.message}`, 'error');
      return null;
    }
  }
  let voiceStatusPending = false;
  async function refreshVoiceStatus() {
    if (voiceStatusPending || !state.status?.engine_loaded) return;
    voiceStatusPending = true;
    try {
      const voice = await api('/api/voice/status');
      $('learnedTtsStatus').textContent = voice.learned_tts_ready ? `Trained checkpoint available · ${shortNumber(voice.tts_steps)} steps` : 'Not ready yet · paired audio/text training required';
      $('learnedTtsBadge').textContent = voice.learned_tts_ready ? 'READY' : 'NOT READY';
      $('learnedTtsBadge').classList.toggle('ready', !!voice.learned_tts_ready);
      $('voiceBackendSelect').querySelector('option[value="learned"]').disabled = !voice.learned_tts_ready;
      const speakerSelect = $('voiceSpeakerSelect'); const savedSpeaker = storage.get('voiceSpeaker', ''); speakerSelect.replaceChildren();
      const noSpeaker = document.createElement('option'); noSpeaker.value = ''; noSpeaker.textContent = voice.speaker_profiles?.length ? 'Choose enrolled voice' : 'No enrolled voice yet'; speakerSelect.appendChild(noSpeaker);
      (voice.speaker_profiles || []).forEach((profile) => { const option = document.createElement('option'); option.value = profile.speaker_id; option.textContent = `${(profile.aliases || []).join(', ') || 'Unnamed speaker'} · ${profile.speaker_id.slice(0, 8)}`; speakerSelect.appendChild(option); (profile.aliases || []).forEach((alias) => { const aliasOption = document.createElement('option'); aliasOption.value = alias; aliasOption.textContent = `${alias} · alias`; speakerSelect.appendChild(aliasOption); }); });
      if ([...speakerSelect.options].some((o) => o.value === savedSpeaker)) speakerSelect.value = savedSpeaker;
    } catch { /* Agent startup may have failed; main status reports it. */ }
    finally { voiceStatusPending = false; }
  }
  function navigate(view) {
    if (!['chat','learning','vision','memory'].includes(view)) return;
    state.activeView = view;
    document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
    document.querySelectorAll('.view-pane').forEach((pane) => pane.classList.toggle('active', pane.dataset.viewPanel === view));
    const titles = {chat:'Conversation', learning:'Learning studio', vision:'Vision & voice', memory:'Memory explorer'};
    $('viewTitle').textContent = titles[view];
    $('contentScroll').scrollTop = 0;
    $('sidebar').classList.remove('open'); $('sidebarScrim').classList.remove('active');
    if (view === 'learning') refreshJobs();
    if (view === 'vision') populateCameras();
    if (view === 'memory') refreshStatus();
  }
  async function loadChats(preferSaved = true) {
    try {
      const result = await api('/api/chats');
      const chats = result.chats || [];
      const list = $('conversationList'); list.replaceChildren();
      if (!chats.length) {
        const empty = document.createElement('div'); empty.className = 'sidebar-empty'; empty.textContent = 'No conversations yet.'; list.appendChild(empty); return;
      }
      chats.forEach((chat) => {
        const button = document.createElement('button');
        button.className = `conversation-item${chat.id === state.currentChatId ? ' active' : ''}`;
        button.dataset.chatId = chat.id;
        button.innerHTML = `<span class="chat-item-icon">◌</span><span class="chat-item-text"><strong>${escapeHTML(chat.title || 'New conversation')}</strong><small>${chat.message_count ? `${shortNumber(chat.message_count)} messages · ${escapeHTML(dateLabel(chat.updated_at))}` : 'No messages yet'}</small></span>`;
        button.addEventListener('click', () => openChat(chat.id));
        list.appendChild(button);
      });
      $('chatCountHint').textContent = `${chats.length} saved conversations`;
      const currentStillExists = chats.some((x) => x.id === state.currentChatId);
      if (!currentStillExists) state.currentChatId = null;
      if (preferSaved && !state.currentChatId) {
        const savedId = storage.get('chatId');
        state.currentChatId = chats.some((c) => c.id === savedId) ? savedId : chats[0].id;
      }
      if (state.currentChatId && !$('messages').dataset.loadedChatId) await openChat(state.currentChatId);
    } catch (error) {
      $('conversationList').innerHTML = `<div class="sidebar-empty">Could not load history: ${escapeHTML(error.message)}</div>`;
    }
  }
  async function createChat() {
    try {
      const chat = await postJSON('/api/chats', {title:'New conversation'});
      state.currentChatId = chat.id; storage.set('chatId', chat.id);
      state.currentChatTitle = chat.title;
      $('messages').dataset.loadedChatId = chat.id;
      $('messages').replaceChildren();
      $('messages').appendChild(buildWelcome());
      $('activeChatTitle').textContent = chat.title;
      $('conversationMeta').textContent = 'Persistent · New conversation';
      $('chatHeading').innerHTML = `A mind in progress<span class="title-period">.</span>`;
      await loadChats(false); navigate('chat'); $('messageInput').focus();
    } catch (error) { toast(`Couldn't create a conversation: ${error.message}`, 'error'); }
  }
  async function openChat(chatId, {navigateToChat = true} = {}) {
    try {
      const chat = await api(`/api/chats/${encodeURIComponent(chatId)}`);
      state.currentChatId = chat.id; state.currentChatTitle = chat.title; storage.set('chatId', chat.id);
      $('messages').dataset.loadedChatId = chat.id;
      $('messages').replaceChildren();
      if (!chat.messages.length) $('messages').appendChild(buildWelcome());
      chat.messages.forEach((message, i) => appendMessage(message, {isLatestAssistant: message.role === 'assistant' && !chat.messages.slice(i+1).some((m) => m.role === 'assistant')}));
      $('activeChatTitle').textContent = chat.title || 'New conversation';
      $('conversationMeta').textContent = `${chat.messages.length} saved messages · Local memory persists across chats`;
      $('chatHeading').innerHTML = `${escapeHTML(chat.title || 'A mind in progress')}<span class="title-period">.</span>`;
      document.querySelectorAll('.conversation-item').forEach((item) => item.classList.toggle('active', item.dataset.chatId === chat.id));
      await loadChats(false);
      scrollMessages(true);
      if (navigateToChat) navigate('chat');
    } catch (error) { toast(`Could not open conversation: ${error.message}`, 'error'); }
  }
  function buildWelcome() {
    if (!welcomeTemplate) return document.createElement('div');
    const node = welcomeTemplate.cloneNode(true);
    node.id = 'welcomeStateClone';
    node.querySelectorAll('[id]').forEach((item) => item.removeAttribute('id'));
    node.querySelectorAll('.suggestion-card').forEach((button) => button.addEventListener('click', () => { $('messageInput').value = button.dataset.prompt || ''; resizeComposer(); $('messageInput').focus(); }));
    return node;
  }
  function appendMessage(message, {isLatestAssistant = false, temporary = false} = {}) {
    const welcome = $('messages').querySelector('.welcome-state');
    if (welcome) welcome.remove();
    const role = message.role || 'assistant';
    const wrapper = document.createElement('div');
    wrapper.className = `message-row ${role}${temporary ? ' temporary-message' : ''}`;
    wrapper.dataset.messageId = message.id || '';
    if (temporary) wrapper.dataset.temporary = 'true';
    const avatar = role === 'user' ? 'M' : role === 'assistant' ? 'A' : 'i';
    const time = message.created_at ? timeLabel(message.created_at) : timeLabel(Date.now()/1000);
    const badge = role === 'assistant' && message.response_mode ? `<span class="message-badge">${escapeHTML(message.response_mode.replaceAll('_',' '))}</span>` : '';
    wrapper.innerHTML = `<div class="message-avatar">${avatar}</div><div class="message-body"><div class="message-bubble">${escapeHTML(message.content || '')}</div><div class="message-meta"><span>${role === 'user' ? 'You' : role === 'assistant' ? 'Aletheia' : 'System'}</span><span>·</span><time>${escapeHTML(time)}</time>${badge}</div>${role === 'assistant' && isLatestAssistant && !temporary ? `<div class="message-actions"><button class="message-action" data-feedback="true" data-positive="true" title="This response was useful">♡ Helpful</button><button class="message-action" data-feedback="true" data-positive="false" title="This response was not useful">↶ Not helpful</button><button class="message-action" data-copy="true" title="Copy response">⧉ Copy</button></div>` : ''}</div>`;
    wrapper.querySelectorAll('[data-feedback]').forEach((button) => button.addEventListener('click', (event) => submitFeedback(message, event.currentTarget.dataset.positive === 'true')));
    wrapper.querySelector('[data-copy]')?.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(String(message.content || '')); toast('Reply copied', 'success', 1500); }
      catch { toast('Clipboard access was blocked by the browser.', 'error'); }
    });
    $('messages').appendChild(wrapper);
    return wrapper;
  }
  async function submitFeedback(message, positive) {
    if (!message.id || !state.currentChatId) return;
    try {
      const result = await postJSON(`/api/chats/${encodeURIComponent(state.currentChatId)}/messages/${encodeURIComponent(message.id)}/feedback`, {positive, note:''});
      const training = result.online_learning?.accepted ? ' A guarded dialogue update was accepted.' : result.online_learning?.reason ? ` (${result.online_learning.reason.replaceAll('_',' ')})` : '';
      toast(`${positive ? 'Thanks for the positive feedback.' : 'Feedback saved.'}${training}`, 'success', 4800);
      await refreshStatus();
    } catch (error) { toast(error.message, 'error', 5000); }
  }
  function scrollMessages(force = false) {
    if (force || state.autoScroll) $('messages').scrollTop = $('messages').scrollHeight;
  }
  function showTyping() {
    const existing = $('typingIndicator'); if (existing) existing.remove();
    const node = document.createElement('div'); node.id = 'typingIndicator'; node.className = 'typing-row';
    node.innerHTML = `<div class="message-avatar">A</div><div class="typing-dots"><span></span><span></span><span></span></div><span>Aletheia is thinking…</span>`;
    $('messages').appendChild(node); scrollMessages();
  }
  function resizeComposer() {
    const input = $('messageInput'); input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  }
  async function sendMessage(value) {
    const message = String(value ?? $('messageInput').value).trim();
    if (!message || state.sending) return;
    if (!state.currentChatId) {
      const created = await api('/api/chats', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:'New conversation'})});
      state.currentChatId = created.id; storage.set('chatId', created.id);
    }
    const activeViewAtSend = state.activeView;
    state.sending = true;
    const input = $('messageInput'); input.value = ''; resizeComposer();
    $('sendButton').disabled = true; $('sendButton').innerHTML = '<span>Thinking</span><span>…</span>';
    const temp = appendMessage({role:'user',content:message}, {temporary:true});
    showTyping(); setOrbState('thinking'); setEngineStatus('Processing message', 'loading');
    try {
      const response = await postJSON(`/api/chats/${encodeURIComponent(state.currentChatId)}/messages`, {message, max_tokens:state.replyTokens});
      $('typingIndicator')?.remove();
      await openChat(state.currentChatId, {navigateToChat: activeViewAtSend === 'chat' && state.activeView === 'chat'});
      if (state.voiceEnabled && response.reply) await speakText(response.reply);
      else setOrbState(state.cameraEnabled ? 'seeing' : 'idle');
      await Promise.all([refreshStatus(), loadChats(false)]);
    } catch (error) {
      $('typingIndicator')?.remove();
      if (temp.isConnected) temp.remove();
      appendMessage({role:'assistant',content:`I couldn't complete that request. ${error.message}\n\nYour message may have been saved in the conversation history; you can retry when the backend is available.`}, {temporary:true});
      toast(`Message failed: ${error.message}`, 'error', 5500);
      setOrbState('idle'); setEngineStatus('Request failed', 'error');
    } finally {
      state.sending = false; $('sendButton').disabled = false; $('sendButton').innerHTML = '<span>Send</span><span>↑</span>';
      input.focus();
    }
  }
  function getSpeechRecognition() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }
  async function prepareMicMeter(stream) {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      state.audioContext = new AudioContextClass();
      const source = state.audioContext.createMediaStreamSource(stream);
      state.analyser = state.audioContext.createAnalyser(); state.analyser.fftSize = 256;
      source.connect(state.analyser);
      const data = new Uint8Array(state.analyser.frequencyBinCount);
      const tick = () => {
        if (!state.analyser || !state.listening) return;
        state.analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (const value of data) { const sample=(value-128)/128; sum += sample*sample; }
        const rms = Math.sqrt(sum/data.length);
        const level = Math.min(1, rms*6); $('orbStage').style.setProperty('--level', level.toFixed(3)); $('orbStage').style.setProperty('--orb-scale', (1 + level*0.16).toFixed(3));
        $('micVisualizer').querySelectorAll('.sound-wave i').forEach((bar, index) => {
          const varied = Math.max(.3, Math.min(2.4, 0.45 + rms*10 + Math.abs(Math.sin(Date.now()/210 + index))*rms*18));
          bar.style.transform = `scaleY(${varied})`;
        });
        state.animFrame = requestAnimationFrame(tick);
      };
      if (state.audioContext.state === 'suspended') await state.audioContext.resume();
      tick();
    } catch (error) { console.debug('Microphone analyser unavailable', error); }
  }
  async function releaseMic() {
    state.listening = false; cancelAnimationFrame(state.animFrame); state.analyser = null;
    if (state.micStream) state.micStream.getTracks().forEach((track) => track.stop());
    state.micStream = null;
    if (state.audioContext) { try { await state.audioContext.close(); } catch {} state.audioContext = null; }
    $('micBadge').textContent = 'OFF'; $('micBadge').classList.remove('enabled');
    $('micVisualizer').classList.remove('active'); $('micVisualLabel').textContent = 'Microphone is idle';
    $('micButton').classList.remove('active'); $('voiceInputToggle').innerHTML = 'Start voice chat <span>♩</span>';
    $('voiceCaptureBar').classList.add('hidden');
    $('orbStage').style.setProperty('--level','.04'); $('orbStage').style.setProperty('--orb-scale','1.006');
    if (state.activity === 'listening') setOrbState('idle');
  }
  async function startVoiceInput() {
    if (state.listening) { stopVoiceInput(); return; }
    const Recognition = getSpeechRecognition();
    if (!Recognition) { toast('Speech recognition is not available in this browser. Try Microsoft Edge or use the text box.', 'error', 5200); return; }
    try {
      state.micStream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
      state.listening = true;
      await prepareMicMeter(state.micStream);
      const recognition = new Recognition(); state.recognition = recognition;
      recognition.lang = $('speechLanguage').value; recognition.continuous = true; recognition.interimResults = true; recognition.maxAlternatives = 1;
      state.voiceFinalText = ''; state.listening = true;
      recognition.onstart = () => {
        $('micBadge').textContent = 'LISTENING'; $('micBadge').classList.add('enabled'); $('micVisualizer').classList.add('active'); $('micVisualLabel').textContent = 'Listening for speech…';
        $('micButton').classList.add('active'); $('voiceInputToggle').innerHTML = 'Stop voice chat <span>■</span>';
        $('voiceCaptureBar').classList.remove('hidden'); $('voiceCaptureText').textContent = 'Listening… say a phrase, then press Stop'; setOrbState('listening');
      };
      recognition.onresult = (event) => {
        let interim = '';
        for (let i=event.resultIndex;i<event.results.length;i++) {
          const transcript = event.results[i][0]?.transcript || '';
          if (event.results[i].isFinal) state.voiceFinalText += `${transcript} `; else interim += transcript;
        }
        $('voiceCaptureText').textContent = (state.voiceFinalText + interim).trim() || 'Listening… speak naturally';
        if (interim) { $('messageInput').value = (state.voiceFinalText + interim).trim(); resizeComposer(); }
      };
      recognition.onerror = (event) => {
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') toast('Microphone or browser speech access was denied.', 'error');
        else if (!['no-speech','aborted'].includes(event.error)) toast(`Speech recognition: ${event.error}`, 'error');
      };
      recognition.onend = async () => {
        const transcript = state.voiceFinalText.trim();
        state.recognition = null;
        await releaseMic();
        if (transcript) { $('messageInput').value = transcript; resizeComposer(); await sendMessage(transcript); }
      };
      recognition.start();
    } catch (error) {
      await releaseMic();
      toast(`Could not start microphone: ${error.message}. Check browser permissions.`, 'error', 5400);
    }
  }
  function stopVoiceInput() {
    try { if (state.recognition) state.recognition.stop(); else releaseMic(); }
    catch { releaseMic(); }
  }
  function populateVoices() {
    if (!('speechSynthesis' in window)) { $('browserVoiceSelect').innerHTML = '<option value="">Speech synthesis unavailable</option>'; $('testVoice').disabled = true; return; }
    state.speechVoices = window.speechSynthesis.getVoices() || [];
    const selected = storage.get('browserVoice', '');
    const select = $('browserVoiceSelect'); select.replaceChildren();
    const defaultOpt = document.createElement('option'); defaultOpt.value=''; defaultOpt.textContent='Default browser voice'; select.appendChild(defaultOpt);
    state.speechVoices.forEach((voice, index) => {
      const opt = document.createElement('option'); opt.value=String(index); opt.textContent=`${voice.name} · ${voice.lang}${voice.default?' · default':''}`; select.appendChild(opt);
    });
    if (selected !== '' && state.speechVoices[Number(selected)]) select.value=selected;
  }
  function speakText(rawText) {
    return new Promise(async (resolve) => {
      const cleanText = String(rawText).replace(/^\[(?:generated|experimental|quoted|not_ready)\]\s*/i, '').trim();
      if (!cleanText) { setOrbState('idle'); resolve(); return; }
      if (state.voiceBackend === 'learned' && state.status?.tts_ready) {
        const alias = $('voiceSpeakerSelect').value || '';
        if (!alias.trim()) { toast('Select an enrolled speaker for learned TTS first.', 'error'); setOrbState('idle'); resolve(); return; }
        try {
          setOrbState('speaking', 'Generating learned voice', 'Aletheia’s trained TTS is rendering audio.');
          const response = await fetch('/api/voice/synthesize', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:cleanText,speaker:alias.trim()})});
          if (!response.ok) { const body = await response.json().catch(()=>({detail:response.statusText})); throw new Error(body.detail || 'Learned voice synthesis failed'); }
          const blob = await response.blob(); const url = URL.createObjectURL(blob); const audio = new Audio(url);
          let finished = false;
          const finish = (showError = false) => {
            if (finished) return; finished = true;
            audio.onended = null; audio.onerror = null;
            URL.revokeObjectURL(url);
            if (state.currentAudio === audio) { state.currentAudio = null; state.currentAudioUrl = null; }
            if (state.speechResolve === resolve) state.speechResolve = null;
            if (showError) toast('The learned audio could not be played.', 'error');
            setOrbState('idle'); resolve();
          };
          state.currentAudio = audio; state.currentAudioUrl = url; state.speechResolve = resolve;
          audio.onended = () => finish();
          audio.onerror = () => finish(true);
          await audio.play(); return;
        } catch (error) { toast(error.message, 'error'); state.voiceBackend='browser'; $('voiceBackendSelect').value='browser'; storage.set('voiceBackend','browser'); }
      }
      if (!('speechSynthesis' in window)) { toast('This browser has no speech synthesis support.', 'error'); setOrbState('idle'); resolve(); return; }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(cleanText);
      const voiceIndex = $('browserVoiceSelect').value;
      if (voiceIndex !== '' && state.speechVoices[Number(voiceIndex)]) utterance.voice = state.speechVoices[Number(voiceIndex)];
      utterance.rate = state.voiceRate; utterance.pitch = state.voicePitch;
      utterance.onstart = () => setOrbState('speaking');
      utterance.onend = () => { setOrbState('idle'); resolve(); };
      utterance.onerror = (event) => { setOrbState('idle'); if (event.error !== 'canceled' && event.error !== 'interrupted') toast(`Speech output failed: ${event.error}`, 'error'); resolve(); };
      setOrbState('speaking'); window.speechSynthesis.speak(utterance);
    });
  }
  function stopSpeaking() {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    document.querySelectorAll('audio').forEach((audio) => { try { audio.pause(); audio.currentTime=0; } catch {} });
    if (state.currentAudioUrl) URL.revokeObjectURL(state.currentAudioUrl);
    state.currentAudio = null; state.currentAudioUrl = null;
    const resolveSpeech = state.speechResolve; state.speechResolve = null;
    if (resolveSpeech) resolveSpeech();
    if (state.activity === 'speaking') setOrbState('idle');
  }
  async function populateCameras() {
    const select=$('cameraDeviceSelect');
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices=await navigator.mediaDevices.enumerateDevices(); const cameras=devices.filter((d)=>d.kind==='videoinput'); const selected=select.value;
      select.replaceChildren(); const base=document.createElement('option');base.value='';base.textContent='Default camera';select.appendChild(base);
      cameras.forEach((cam,i)=>{const opt=document.createElement('option');opt.value=cam.deviceId;opt.textContent=cam.label||`Camera ${i+1}`;select.appendChild(opt);});
      if ([...select.options].some((o)=>o.value===selected)) select.value=selected;
    } catch { /* Device listing is optional until permission is granted. */ }
  }
  async function startCamera() {
    if (state.cameraStream) { stopCamera(); return; }
    if (!navigator.mediaDevices?.getUserMedia) { toast('Camera access is not supported in this browser.', 'error'); return; }
    try {
      const selected=$('cameraDeviceSelect').value;
      state.cameraStream=await navigator.mediaDevices.getUserMedia({video:selected?{deviceId:{exact:selected},width:{ideal:1280},height:{ideal:720}}:{width:{ideal:1280},height:{ideal:720}},audio:false});
      const video=$('cameraVideo'); video.srcObject=state.cameraStream; await video.play();
      state.cameraEnabled=true; $('cameraStage').classList.add('active'); $('cameraPlaceholder').classList.add('hidden'); $('cameraLiveIndicator').classList.remove('hidden'); $('cameraBadge').textContent='ON'; $('cameraBadge').classList.add('enabled');
      $('cameraToggle').innerHTML='Disable camera <span>×</span>'; $('cameraSnapshot').disabled=false;
      await populateCameras(); setOrbState('seeing');
      state.cameraTimer=window.setInterval(()=>{if(state.cameraEnabled&&!state.sending&&!state.cameraBusy)observeCameraFrame(true);},8000);
      await observeCameraFrame(false);
      toast('Camera enabled. Snapshots are observed locally and then discarded.', 'success');
    } catch (error) {
      stopCamera(); toast(`Could not start camera: ${error.message}. Check browser permissions.`, 'error', 5000);
    }
  }
  function stopCamera() {
    state.cameraEnabled=false; if(state.cameraTimer)clearInterval(state.cameraTimer);state.cameraTimer=0;
    if(state.cameraStream)state.cameraStream.getTracks().forEach((track)=>track.stop()); state.cameraStream=null;
    const video=$('cameraVideo'); if(video) video.srcObject=null;
    $('cameraStage').classList.remove('active');$('cameraPlaceholder').classList.remove('hidden');$('cameraLiveIndicator').classList.add('hidden');$('cameraBadge').textContent='OFF';$('cameraBadge').classList.remove('enabled');$('cameraToggle').innerHTML='Enable camera <span>↗</span>';$('cameraSnapshot').disabled=true;
    if(state.activity==='seeing')setOrbState('idle');
  }
  function canvasToBlob(canvas,type='image/jpeg',quality=.82){return new Promise((resolve)=>canvas.toBlob(resolve,type,quality));}
  async function observeCameraFrame(silent=false) {
    if(!state.cameraStream||state.cameraBusy)return;
    const video=$('cameraVideo');if(video.readyState<2||!video.videoWidth)return;
    state.cameraBusy=true;
    try {
      const canvas=document.createElement('canvas'); const maxWidth=960; const scale=Math.min(1,maxWidth/video.videoWidth);canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
      const context=canvas.getContext('2d');context.drawImage(video,0,0,canvas.width,canvas.height);
      const blob=await canvasToBlob(canvas);if(!blob)throw new Error('Browser could not encode the frame');
      const form=new FormData();form.append('file',blob,`camera-${Date.now()}.jpg`);
      if(!silent)setOrbState('seeing');
      const result=await api('/api/perception/image',{method:'POST',body:form});
      const obs=result.observation||{};
      $('visionResult').innerHTML=`<div class="result-icon">◈</div><div><strong>Visual observation recorded</strong><p>Entity ${escapeHTML(obs.entity_id||'—')} · ${Number(obs.object_count||0)} latent objects · novelty ${Number(obs.novelty||0).toFixed(3)}. This is a learned visual representation, not an image caption.</p></div>`;
      if(!silent)toast('Aletheia observed a camera frame.', 'success', 1800);
      if(!state.sending&&!state.listening&&state.activity==='seeing')setOrbState('idle');
      refreshStatus();
    } catch(error) { if(!silent)toast(`Camera observation failed: ${error.message}`,'error'); }
    finally {state.cameraBusy=false;}
  }
  function writeWavMono(samples, sampleRate) {
    const bytesPerSample=2;const buffer=new ArrayBuffer(44+samples.length*bytesPerSample);const view=new DataView(buffer);
    const writeString=(offset,value)=>{for(let i=0;i<value.length;i++)view.setUint8(offset+i,value.charCodeAt(i));};
    writeString(0,'RIFF');view.setUint32(4,36+samples.length*2,true);writeString(8,'WAVE');writeString(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);writeString(36,'data');view.setUint32(40,samples.length*2,true);
    let offset=44;for(const sample of samples){const s=Math.max(-1,Math.min(1,sample));view.setInt16(offset,s<0?s*0x8000:s*0x7fff,true);offset+=2;}
    return new Blob([buffer],{type:'audio/wav'});
  }
  async function recordAudioObservation() {
    if(!navigator.mediaDevices?.getUserMedia){toast('Microphone is not supported in this browser.','error');return;}
    $('audioObserveButton').disabled=true;$('audioObserveButton').textContent='Recording…';setOrbState('listening');
    let stream, recorder, context;
    try {
      stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
      recorder=new MediaRecorder(stream);const parts=[];
      recorder.ondataavailable=(event)=>{if(event.data.size)parts.push(event.data);};
      const stopped=new Promise((resolve,reject)=>{recorder.onerror=()=>reject(new Error('Browser recorder failed'));recorder.onstop=resolve;});
      recorder.start();await new Promise((resolve)=>setTimeout(resolve,3000));recorder.stop();await stopped;
      const blob=new Blob(parts,{type:recorder.mimeType||'audio/webm'});const raw=await blob.arrayBuffer();
      const AudioContextClass=window.AudioContext||window.webkitAudioContext;if(!AudioContextClass)throw new Error('Web Audio is unavailable');
      context=new AudioContextClass();const decoded=await context.decodeAudioData(raw.slice(0));
      const targetRate=16000;const targetLength=Math.max(1,Math.round(decoded.duration*targetRate));
      const offline=new OfflineAudioContext(1,targetLength,targetRate);const source=offline.createBufferSource();source.buffer=decoded;source.connect(offline.destination);source.start();const resampled=await offline.startRendering();const samples=resampled.getChannelData(0);
      const wav=writeWavMono(samples,targetRate);const form=new FormData();form.append('file',wav,`microphone-${Date.now()}.wav`);
      setOrbState('seeing','Encoding audio','Aletheia is encoding the short audio sample; it will not be stored as raw audio.');
      const result=await api('/api/perception/audio',{method:'POST',body:form});
      $('micVisualLabel').textContent=`Audio encoded · ${result.observation.embedding_dim} dimensions`;
      toast('Audio observation recorded. It was not transcribed or stored as a raw recording.','success',3500);await refreshStatus();
    } catch(error) {toast(`Audio observation failed: ${error.message}`,'error',5000);}
    finally {if(stream)stream.getTracks().forEach((track)=>track.stop());if(context)try{await context.close();}catch{}$('audioObserveButton').disabled=false;$('audioObserveButton').textContent='Record 3s observation';if(state.activity!=='speaking')setOrbState('idle');}
  }
  async function enrollVoice() {
    const file=$('voiceSampleFile').files?.[0];const alias=$('voiceAlias').value.trim();
    if(!file){toast('Choose a WAV voice sample first.','error');return;}if(!alias){toast('Enter a label for the voice sample.','error');return;}if(!$('voiceConsent').checked){toast('Confirm that you have permission to use this voice sample.','error');return;}
    const form=new FormData();form.append('file',file);form.append('alias',alias);form.append('consent_confirmed','true');$('enrollVoiceButton').disabled=true;$('enrollVoiceButton').textContent='Enrolling…';
    try {const result=await api('/api/voice/enroll',{method:'POST',body:form});toast(`Voice identity enrolled: ${result.speaker.alias||result.speaker.speaker_id||alias}. ${result.note}`,'success',6000);$('voiceSampleFile').value='';await refreshStatus();await refreshVoiceStatus();}
    catch(error){toast(error.message,'error',6000);}finally{$('enrollVoiceButton').disabled=false;$('enrollVoiceButton').textContent='Enroll';}
  }
  async function uploadLearningFile(file) {
    const suffix='.'+(file.name.split('.').pop()||'').toLowerCase();
    if(!supportedFiles.includes(suffix)){toast(`${file.name}: unsupported format. Supported: ${supportedFiles.join(', ')}`,'error',6000);return;}
    if(file.size>50*1024*1024){toast(`${file.name}: larger than the 50 MiB upload limit.`,'error');return;}
    const form=new FormData();form.append('file',file);form.append('license_name',$('licenseInput').value.trim()||'unspecified');form.append('title',file.name);form.append('auto_train',String($('autoTrainCheckbox').checked));
    showUploadPreview(file.name,'Uploading…');setOrbState('learning');
    try {const result=await api('/api/learning/upload',{method:'POST',body:form});const job=result.job||{};toast(`${file.name}: ${result.message||'queued'}`,job.state==='failed'?'error':'success',4500);$('uploadQueuePreview').classList.add('hidden');await refreshJobs();await refreshStatus();}
    catch(error){toast(`${file.name} was not queued: ${error.message}`,'error',6000);$('uploadQueuePreview').classList.add('hidden');}
    finally {if(state.activity==='learning'&&!state.jobs.some((j)=>['queued','extracting','splitting','indexing','training','evaluating'].includes(j.state)))setOrbState('idle');}
  }
  function showUploadPreview(name,status) {const preview=$('uploadQueuePreview');preview.classList.remove('hidden');preview.innerHTML=`<div class="upload-preview-item"><span>◈</span><strong>${escapeHTML(name)}</strong><span style="margin-left:auto;color:var(--muted)">${escapeHTML(status)}</span></div>`;}
  async function refreshJobs() {
    try {
      const data=await api('/api/learning/jobs?limit=30');state.jobs=data.jobs||[];
      $('jobTotalMetric').textContent=String(state.jobs.length);$('jobAcceptedMetric').textContent=String(state.jobs.filter((j)=>j.state==='accepted').length);$('jobActiveMetric').textContent=String(state.jobs.filter((j)=>['queued','extracting','splitting','indexing','training','evaluating'].includes(j.state)).length);
      const active=state.jobs.filter((j)=>['queued','extracting','splitting','indexing','training','evaluating'].includes(j.state)).length;
      $('learningNavCount').textContent=active;$('learningNavCount').classList.toggle('hidden',active===0);
      const list=$('jobsList');list.replaceChildren();
      if(!state.jobs.length){const empty=document.createElement('div');empty.className='empty-jobs';empty.textContent='Your learning jobs will appear here.';list.appendChild(empty);}
      state.jobs.forEach((job)=>{
        const stateName=String(job.state||'unknown');const card=document.createElement('div');card.className='job-card';
        const percent=Math.max(0,Math.min(100,Number(job.progress)||0));const detail=job.status_message||job.error||'Queued';
        const reason=job.error||job.metrics?.decision_reason||'';
        card.innerHTML=`<div class="job-card-top"><div class="job-file"><div class="job-file-icon">${['accepted'].includes(stateName)?'✓':(['failed','rejected'].includes(stateName)?'!':'↥')}</div><div><strong>${escapeHTML(job.source_name||job.provenance?.title||'Learning material')}</strong><small>${escapeHTML(dateLabel(job.created_at))} · ${shortNumber(job.source_bytes)} bytes</small></div></div><span class="job-state ${escapeHTML(stateName)}">${escapeHTML(stateName)}</span></div><div class="job-progress"><span style="width:${percent}%"></span></div><div class="job-detail"><span>${escapeHTML(detail)}</span><strong>${percent}%</strong></div>${reason?`<p class="job-reason">${escapeHTML(reason)}</p>`:''}${stateName==='queued'?`<div class="job-actions"><button class="soft-button" data-run-job="${escapeHTML(job.id)}">Run now</button></div>`:''}${['failed','rejected'].includes(stateName)?`<div class="job-actions"><button class="soft-button" data-retry-job="${escapeHTML(job.id)}">Retry guarded job</button>${job.report_path?`<button class="soft-button" data-view-report="${escapeHTML(job.id)}">View report</button>`:''}</div>`:''}`;
        card.querySelector('[data-retry-job]')?.addEventListener('click',()=>retryJob(job.id));card.querySelector('[data-run-job]')?.addEventListener('click',()=>runJobNow(job.id));card.querySelector('[data-view-report]')?.addEventListener('click',()=>viewJobDetails(job.id));list.appendChild(card);
      });
      if(active>0&&state.activity!=='thinking'&&state.activity!=='speaking'&&state.activity!=='listening')setOrbState('learning');else if(active===0&&state.activity==='learning')setOrbState('idle');
    } catch(error) { $('jobsList').innerHTML=`<div class="empty-jobs">Could not load jobs: ${escapeHTML(error.message)}</div>`; }
  }
  async function retryJob(jobId) {try{const result=await api(`/api/learning/jobs/${encodeURIComponent(jobId)}/retry`,{method:'POST'});toast(`Job requeued: ${jobId}`, 'success');await refreshJobs();}catch(error){toast(`Could not retry job: ${error.message}`,'error');}}
  async function viewJobDetails(jobId) {try{const response=await fetch(`/api/learning/jobs/${encodeURIComponent(jobId)}/report`);if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(body.detail||'Report is unavailable');}downloadBlob(await response.blob(),`aletheia-learning-${jobId}.json`);}catch(error){toast(error.message,'error');}}
  async function runJobNow(jobId) {try{await api(`/api/learning/jobs/${encodeURIComponent(jobId)}/run`,{method:'POST'});toast('Learning worker enabled. The job will run through the guarded pipeline.','success');setOrbState('learning');await refreshJobs();}catch(error){toast(error.message,'error');}}
  async function submitTextLearning(event) {
    event.preventDefault();const text=$('textLessonBody').value.trim();if(text.length<40){toast('Please provide at least 40 characters of teaching material.','error');return;}
    const button=$('submitTextLesson');button.disabled=true;button.textContent='Queueing…';
    try{const result=await postJSON('/api/learning/text',{text,title:$('textLessonTitle').value.trim()||'Pasted lesson',license:$('textLessonLicense').value.trim()||'user-provided',auto_train:$('autoTrainCheckbox').checked});toast(`Text material queued: ${result.job.id}`,'success');$('textLessonBody').value='';updateLessonCount();await refreshJobs();setOrbState('learning');}
    catch(error){toast(`Couldn't queue text lesson: ${error.message}`,'error',6000);}finally{button.disabled=false;button.innerHTML='Queue & teach <span>↗</span>';}
  }
  async function searchMemory(event) {
    event.preventDefault();const query=$('memorySearchInput').value.trim();if(!query)return;
    const container=$('memoryResults');container.innerHTML='<div class="empty-jobs">Searching Aletheia’s indexed memory…</div>';
    try{const data=await api(`/api/memory/search?q=${encodeURIComponent(query)}&limit=12`);container.replaceChildren();if(!data.results?.length){container.innerHTML='<div class="memory-empty"><div>⌕</div><strong>No matching memories found</strong><p>This does not mean the topic never appeared in conversation; it may not be indexed for search.</p></div>';return;}
      data.results.forEach((row)=>{const card=document.createElement('article');card.className='memory-result-card';card.innerHTML=`<div class="memory-result-head"><strong>${escapeHTML(row.title||'Untitled memory')}</strong><span class="memory-result-kind">${escapeHTML(row.kind||'memory')}</span></div><div class="memory-result-meta">${escapeHTML(row.source||'local memory')} · ${escapeHTML(dateLabel(row.created||0))}${row.accepted===false?' · rejected':''}</div><div class="memory-result-text">${escapeHTML(row.text||'')}</div>`;container.appendChild(card);});}
    catch(error){container.innerHTML=`<div class="empty-jobs">Memory search failed: ${escapeHTML(error.message)}</div>`;}
  }
  function downloadBlob(blob, filename) {const url=URL.createObjectURL(blob);const anchor=document.createElement('a');anchor.href=url;anchor.download=filename;document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);}
  async function exportActiveChat() {if(!state.currentChatId){toast('No conversation selected.','error');return;}try{const chat=await api(`/api/chats/${encodeURIComponent(state.currentChatId)}`);const text=[`# ${chat.title}`,`Exported ${new Date().toLocaleString()}`,'',...chat.messages.map((m)=>`## ${m.role==='user'?'You':m.role==='assistant'?'Aletheia':'System'} — ${dateLabel(m.created_at)}\n\n${m.content}\n`)].join('\n');downloadBlob(new Blob([text],{type:'text/markdown'}),`aletheia-${(chat.title||'conversation').replace(/[^a-z0-9-_]+/gi,'-').slice(0,50)}.md`);toast('Conversation exported.','success',1500);}catch(error){toast(error.message,'error');}}
  async function exportAllData() {try{const chats=await api('/api/chats');const payload={exported_at:new Date().toISOString(),version:'0.34.0',chats:[]};for(const item of chats.chats||[]){payload.chats.push(await api(`/api/chats/${encodeURIComponent(item.id)}`));}downloadBlob(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),'aletheia-conversations.json');toast('Conversation history exported as JSON.','success');}catch(error){toast(error.message,'error');}}
  function setTheme(theme) {const value=theme==='light'?'light':'dark';document.documentElement.dataset.theme=value;storage.set('theme',value);$('themeIcon').textContent=value==='dark'?'☼':'☾';}
  function updateLessonCount() {$('lessonCharCount').textContent=`${$('textLessonBody').value.length.toLocaleString()} characters`;}

  // Sidebar, navigation and chat lifecycle
  document.querySelectorAll('.nav-item').forEach((button)=>button.addEventListener('click',()=>navigate(button.dataset.view)));
  document.querySelectorAll('[data-go-view]').forEach((button)=>button.addEventListener('click',()=>navigate(button.dataset.goView)));
  $('newChat').addEventListener('click',createChat);$('sidebarClose').addEventListener('click',()=>{$('sidebar').classList.remove('open');$('sidebarScrim').classList.remove('active');});
  $('menuToggle').addEventListener('click',()=>{$('sidebar').classList.add('open');$('sidebarScrim').classList.add('active');});$('sidebarScrim').addEventListener('click',()=>{$('sidebar').classList.remove('open');$('sidebarScrim').classList.remove('active');});
  $('refreshChats').addEventListener('click',()=>loadChats(false));$('refreshStatus').addEventListener('click',()=>refreshStatus({quiet:false}));$('refreshJobs').addEventListener('click',refreshJobs);$('scrollToBottom').addEventListener('click',()=>scrollMessages(true));
  $('chatForm').addEventListener('submit',(event)=>{event.preventDefault();sendMessage();});
  $('messageInput').addEventListener('input',resizeComposer);$('messageInput').addEventListener('keydown',(event)=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();sendMessage();}});
  document.querySelectorAll('.suggestion-card').forEach((button)=>button.addEventListener('click',()=>{$('messageInput').value=button.dataset.prompt||'';resizeComposer();$('messageInput').focus();}));
  $('micButton').addEventListener('click',startVoiceInput);$('voiceInputToggle').addEventListener('click',startVoiceInput);$('stopListening').addEventListener('click',stopVoiceInput);
  $('speakReplies').checked=state.voiceEnabled;$('speakReplies').addEventListener('change',()=>{state.voiceEnabled=$('speakReplies').checked;storage.set('speakReplies',state.voiceEnabled);toast(state.voiceEnabled?'Voice replies enabled.':'Voice replies disabled.', 'info',1600);});
  $('orbStage').addEventListener('click',()=>navigate('chat'));
  $('exportChat').addEventListener('click',exportActiveChat);
  document.addEventListener('keydown',(event)=>{if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();createChat();}if(event.key==='Escape'){$('settingsModal').classList.add('hidden');$('sidebar').classList.remove('open');$('sidebarScrim').classList.remove('active');}});

  // Theme and settings
  setTheme(storage.get('theme','dark'));$('themeToggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
  $('settingsButton').addEventListener('click',()=>{$('settingsModal').classList.remove('hidden');});$('closeSettings').addEventListener('click',()=>{$('settingsModal').classList.add('hidden');});$('closeSettingsDone').addEventListener('click',()=>{$('settingsModal').classList.add('hidden');});$('settingsModal').addEventListener('click',(event)=>{if(event.target===$('settingsModal'))$('settingsModal').classList.add('hidden');});
  $('maxReplyTokens').value=String(state.replyTokens);$('maxReplyLabel').textContent=`${state.replyTokens} tokens`;$('tokenHint').textContent=`${state.replyTokens} max reply tokens`;
  $('maxReplyTokens').addEventListener('input',()=>{state.replyTokens=Number($('maxReplyTokens').value);$('maxReplyLabel').textContent=`${state.replyTokens} tokens`;$('tokenHint').textContent=`${state.replyTokens} max reply tokens`;storage.set('replyTokens',state.replyTokens);});
  $('autoScrollCheckbox').checked=state.autoScroll;$('autoScrollCheckbox').addEventListener('change',()=>{state.autoScroll=$('autoScrollCheckbox').checked;storage.set('autoScroll',state.autoScroll);});$('downloadDataButton').addEventListener('click',exportAllData);

  // Learning studio: files, drag/drop, text jobs and progress.
  $('browseFiles').addEventListener('click',(event)=>{event.stopPropagation();$('learningFileInput').click();});$('dropZone').addEventListener('click',(event)=>{if(!event.target.closest('button'))$('learningFileInput').click();});$('dropZone').addEventListener('keydown',(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('learningFileInput').click();}});
  $('learningFileInput').addEventListener('change',async(event)=>{const files=[...(event.target.files||[])];if(files.length){for(const file of files)await uploadLearningFile(file);}event.target.value='';});
  ['dragenter','dragover'].forEach((name)=>$('dropZone').addEventListener(name,(event)=>{event.preventDefault();$('dropZone').classList.add('dragging');}));['dragleave','drop'].forEach((name)=>$('dropZone').addEventListener(name,(event)=>{event.preventDefault();$('dropZone').classList.remove('dragging');}));
  $('dropZone').addEventListener('drop',async(event)=>{const files=[...(event.dataTransfer?.files||[])];if(!files.length){toast('No files were found in that drop.','error');return;}for(const file of files)await uploadLearningFile(file);});
  $('textLessonBody').addEventListener('input',updateLessonCount);$('textLearningForm').addEventListener('submit',submitTextLearning);
  $('autoTrainCheckbox').checked=state.autoTrain;
  $('autoTrainCheckbox').addEventListener('change',()=>{state.autoTrain=$('autoTrainCheckbox').checked;storage.set('autoTrain',state.autoTrain);});

  // Camera and microphone are opt-in. Browser owns the actual device permissions.
  $('cameraToggle').addEventListener('click',startCamera);$('cameraSnapshot').addEventListener('click',()=>observeCameraFrame(false));$('cameraDeviceSelect').addEventListener('change',()=>{if(state.cameraStream){stopCamera();startCamera();}});$('audioObserveButton').addEventListener('click',recordAudioObservation);$('enrollVoiceButton').addEventListener('click',enrollVoice);
  $('speechLanguage').value=storage.get('speechLanguage','en-IN');$('speechLanguage').addEventListener('change',()=>storage.set('speechLanguage',$('speechLanguage').value));
  $('voiceBackendSelect').value=state.voiceBackend;$('voiceBackendSelect').addEventListener('change',()=>{state.voiceBackend=$('voiceBackendSelect').value;storage.set('voiceBackend',state.voiceBackend);});
  $('voiceSpeakerSelect').value=storage.get('voiceSpeaker','');$('voiceSpeakerSelect').addEventListener('change',()=>storage.set('voiceSpeaker',$('voiceSpeakerSelect').value));
  $('voiceRate').value=String(state.voiceRate);$('voiceRateValue').textContent=`${state.voiceRate.toFixed(2).replace(/0$/,'')}×`;$('voiceRate').addEventListener('input',()=>{state.voiceRate=Number($('voiceRate').value);$('voiceRateValue').textContent=`${state.voiceRate.toFixed(2).replace(/0$/,'')}×`;storage.set('voiceRate',state.voiceRate);});
  $('voicePitch').value=String(state.voicePitch);$('voicePitchValue').textContent=state.voicePitch.toFixed(1);$('voicePitch').addEventListener('input',()=>{state.voicePitch=Number($('voicePitch').value);$('voicePitchValue').textContent=state.voicePitch.toFixed(1);storage.set('voicePitch',state.voicePitch);});
  $('testVoice').addEventListener('click',()=>speakText('Hello. I am Aletheia, a learning system in development.'));$('stopSpeaking').addEventListener('click',stopSpeaking);
  $('voiceConsent').checked=false;
  $('memorySearchForm').addEventListener('submit',searchMemory);
  if('speechSynthesis' in window){populateVoices();window.speechSynthesis.onvoiceschanged=populateVoices;}

  // Recover UI state from the backend and keep visible job progress fresh.
  (async()=>{
    try {await api('/api/health');} catch { setEngineStatus('Backend unavailable','error'); toast('Aletheia’s API is not responding. Start it with: python -m aletheia.webapp --root .','error',9000); return; }
    await loadChats(true);
    await refreshStatus();await refreshJobs();populateCameras();
    state.jobsTimer=window.setInterval(()=>{if(state.activeView==='learning'||state.jobs.some((j)=>['queued','extracting','splitting','indexing','training','evaluating'].includes(j.state)))refreshJobs();},1800);
    state.statusTimer=window.setInterval(()=>refreshStatus(),12000);
    if(state.activeView==='chat')$('messageInput').focus({preventScroll:true});
  })();
  window.addEventListener('beforeunload',()=>{if(state.cameraTimer)clearInterval(state.cameraTimer);if(state.cameraStream)state.cameraStream.getTracks().forEach((t)=>t.stop());if(state.micStream)state.micStream.getTracks().forEach((t)=>t.stop());if(state.recognition)try{state.recognition.abort();}catch{}if(state.audioContext)try{state.audioContext.close();}catch{};});
})();
