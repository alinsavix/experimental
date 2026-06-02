// ── Configuration ────────────────────────────────────────────────────
const API_BASE = location.origin + '/1';
const WS_BASE = location.origin.replace(/^http/, 'ws');
const CHANNEL_ID = 'default';

const headers = () => ({
    'Content-Type': 'application/json',
    'Nightbot-Channel': CHANNEL_ID,
});

// ── State ────────────────────────────────────────────────────────────
let state = {
    queue: [],
    currentSong: null,
    playlist: [],
    playlistTotal: 0,
    playlistOffset: 0,
    settings: null,
    isPlaying: false,
    isMuted: false,
    volume: 50,
    history: [],
    playerReady: false,
};

let ytPlayer = null;
let seekInterval = null;

// ── API Helpers ──────────────────────────────────────────────────────
async function api(method, path, body = null) {
    const opts = { method, headers: headers() };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(API_BASE + path, opts);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ message: resp.statusText }));
        throw new Error(err.detail || err.message || 'API error');
    }
    return resp.json();
}

// ── YouTube Player ───────────────────────────────────────────────────
window.onYouTubeIframeAPIReady = function () {
    ytPlayer = new YT.Player('player', {
        width: '100%',
        height: '100%',
        playerVars: {
            autoplay: 0,
            controls: 0,
            disablekb: 1,
            modestbranding: 1,
            rel: 0,
            fs: 0,
        },
        events: {
            onReady: onPlayerReady,
            onStateChange: onPlayerStateChange,
            onError: onPlayerError,
        },
    });
};

function onPlayerReady() {
    state.playerReady = true;
    ytPlayer.setVolume(state.volume);
    if (state.currentSong && state.currentSong.track.provider === 'youtube') {
        loadYouTubeVideo(state.currentSong.track.providerId);
    }
}

function onPlayerStateChange(event) {
    switch (event.data) {
        case YT.PlayerState.PLAYING:
            state.isPlaying = true;
            updatePlayPauseButton();
            startSeekUpdater();
            break;
        case YT.PlayerState.PAUSED:
            state.isPlaying = false;
            updatePlayPauseButton();
            stopSeekUpdater();
            break;
        case YT.PlayerState.ENDED:
            state.isPlaying = false;
            updatePlayPauseButton();
            stopSeekUpdater();
            skipToNext();
            break;
        case YT.PlayerState.BUFFERING:
            break;
    }
}

function onPlayerError(event) {
    const codes = { 2: 'Invalid Video ID', 5: 'HTML5 error', 100: 'Not found', 101: 'Not embeddable', 150: 'Not embeddable' };
    showToast(codes[event.data] || 'Player error', 'error');
    skipToNext();
}

function loadYouTubeVideo(videoId) {
    if (!state.playerReady || !ytPlayer) return;
    document.getElementById('playerPlaceholder').style.display = 'none';
    ytPlayer.loadVideoById(videoId);
    ytPlayer.setVolume(state.isMuted ? 0 : state.volume);
}

function stopPlayer() {
    if (ytPlayer && state.playerReady) {
        ytPlayer.stopVideo();
    }
    state.isPlaying = false;
    updatePlayPauseButton();
    stopSeekUpdater();
    document.getElementById('playerPlaceholder').style.display = 'flex';
}

// ── Seek bar ─────────────────────────────────────────────────────────
function startSeekUpdater() {
    stopSeekUpdater();
    seekInterval = setInterval(updateSeekBar, 500);
}

function stopSeekUpdater() {
    if (seekInterval) { clearInterval(seekInterval); seekInterval = null; }
}

function updateSeekBar() {
    if (!ytPlayer || !state.playerReady) return;
    const current = ytPlayer.getCurrentTime() || 0;
    const duration = ytPlayer.getDuration() || 0;
    document.getElementById('seekBar').max = duration;
    document.getElementById('seekBar').value = current;
    document.getElementById('currentTime').textContent = formatDuration(current);
    document.getElementById('totalTime').textContent = formatDuration(duration);
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// ── UI Updates ───────────────────────────────────────────────────────
function updatePlayPauseButton() {
    document.getElementById('playPauseBtn').textContent = state.isPlaying ? '⏸' : '▶';
}

function updateNowPlaying() {
    const song = state.currentSong;
    if (song) {
        document.getElementById('nowPlayingTitle').textContent = song.track.title;
        document.getElementById('nowPlayingArtist').textContent = song.track.artist + ' • ' + formatDuration(song.track.duration);
        document.title = `▶ ${song.track.title} — Song Requests`;
    } else {
        document.getElementById('nowPlayingTitle').textContent = 'No song playing';
        document.getElementById('nowPlayingArtist').textContent = '';
        document.getElementById('currentTime').textContent = '0:00';
        document.getElementById('totalTime').textContent = '0:00';
        document.getElementById('seekBar').value = 0;
        document.title = 'Song Requests';
    }
}

function updateVolumeUI() {
    document.getElementById('volumeSlider').value = state.volume;
    document.getElementById('muteBtn').textContent = state.isMuted ? '🔇' : (state.volume > 50 ? '🔊' : state.volume > 0 ? '🔉' : '🔇');
    if (state.settings) {
        document.getElementById('settingVolume').value = state.volume;
        document.getElementById('settingVolumeLabel').textContent = state.volume;
    }
}

// ── Queue Rendering ──────────────────────────────────────────────────
function renderQueue() {
    const container = document.getElementById('queueList');
    const searchTerm = document.getElementById('queueSearch').value.toLowerCase();

    let items = state.queue;
    if (searchTerm) {
        items = items.filter(s =>
            (s.track.title || '').toLowerCase().includes(searchTerm) ||
            (s.track.artist || '').toLowerCase().includes(searchTerm) ||
            (s.user?.displayName || '').toLowerCase().includes(searchTerm)
        );
    }

    if (items.length === 0) {
        container.innerHTML = '<div class="empty-state">' + (state.queue.length === 0 ? 'Queue is empty. Request a song to get started!' : 'No matching songs.') + '</div>';
        return;
    }

    container.innerHTML = items.map((song, idx) => songItemHTML(song, idx, true)).join('');

    // Add drag handlers
    setupDragAndDrop(container, state.queue, async (newOrder) => {
        state.queue = newOrder;
        renderQueue();
        await api('PATCH', '/song_requests/queue/order', { order: newOrder.map(s => s._id) });
    });
}

function renderPlaylist() {
    const container = document.getElementById('playlistList');
    const searchTerm = document.getElementById('playlistSearch')?.value.toLowerCase() || '';

    let items = state.playlist;
    if (searchTerm) {
        items = items.filter(s =>
            (s.track.title || '').toLowerCase().includes(searchTerm) ||
            (s.track.artist || '').toLowerCase().includes(searchTerm)
        );
    }

    if (items.length === 0) {
        container.innerHTML = '<div class="empty-state">Playlist is empty.</div>';
    } else {
        container.innerHTML = items.map((song, idx) => songItemHTML(song, idx, false)).join('');
    }

    // Pagination
    const pagDiv = document.getElementById('playlistPagination');
    if (state.playlistTotal > 20) {
        const pages = Math.ceil(state.playlistTotal / 20);
        const currentPage = Math.floor(state.playlistOffset / 20);
        let html = '';
        for (let i = 0; i < pages; i++) {
            html += `<button class="btn btn-sm ${i === currentPage ? 'btn-primary' : 'btn-secondary'}" onclick="loadPlaylistPage(${i})">${i + 1}</button>`;
        }
        pagDiv.innerHTML = html;
    } else {
        pagDiv.innerHTML = '';
    }
}

function songItemHTML(song, idx, isQueue) {
    const thumbnail = song.track.provider === 'youtube'
        ? `<img class="song-thumbnail" src="https://i.ytimg.com/vi/${song.track.providerId}/default.jpg" alt="" />`
        : `<div class="song-thumbnail">🎵</div>`;

    const requester = song.user ? song.user.displayName : 'Playlist';
    const duration = formatDuration(song.track.duration);

    const actions = isQueue
        ? `<div class="song-actions">
        <button class="btn btn-sm btn-secondary" onclick="playItem('${song._id}')" title="Play now">▶</button>
        <button class="btn btn-sm btn-secondary" onclick="promoteItem('${song._id}')" title="Promote">⬆</button>
        <button class="btn btn-sm btn-danger" onclick="deleteQueueItem('${song._id}')" title="Remove">✕</button>
      </div>`
        : `<div class="song-actions">
        <button class="btn btn-sm btn-secondary" onclick="queueFromPlaylist('${song._id}')" title="Add to queue">+Q</button>
        <button class="btn btn-sm btn-danger" onclick="deletePlaylistItem('${song._id}')" title="Remove">✕</button>
      </div>`;

    return `
    <div class="song-item" data-id="${song._id}" draggable="${isQueue}">
      ${isQueue ? '<span class="song-drag-handle">⠿</span>' : ''}
      ${thumbnail}
      <div class="song-info">
        <div class="song-title"><a href="${song.track.url}" target="_blank" rel="noopener">${escapeHtml(song.track.title)}</a></div>
        <div class="song-meta">${escapeHtml(song.track.artist)} • ${duration}</div>
      </div>
      <div class="song-requester">${escapeHtml(requester)}</div>
      ${actions}
    </div>`;
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Drag and Drop ────────────────────────────────────────────────────
function setupDragAndDrop(container, items, onReorder) {
    let dragIdx = null;

    container.querySelectorAll('.song-item[draggable="true"]').forEach((el, idx) => {
        el.addEventListener('dragstart', (e) => {
            dragIdx = idx;
            el.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
        });
        el.addEventListener('dragend', () => {
            el.classList.remove('dragging');
            dragIdx = null;
        });
        el.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });
        el.addEventListener('drop', (e) => {
            e.preventDefault();
            if (dragIdx === null || dragIdx === idx) return;
            const newItems = [...items];
            const [moved] = newItems.splice(dragIdx, 1);
            newItems.splice(idx, 0, moved);
            onReorder(newItems);
        });
    });
}

// ── Settings ─────────────────────────────────────────────────────────
function loadSettingsUI() {
    if (!state.settings) return;
    const s = state.settings;
    document.getElementById('settingEnabled').checked = s.enabled;
    document.getElementById('enableToggle').checked = s.enabled;
    document.getElementById('settingVolume').value = s.volume;
    document.getElementById('settingVolumeLabel').textContent = s.volume;
    document.getElementById('settingSearchProvider').value = s.searchProvider;
    document.getElementById('settingUserLevel').value = s.userLevel;
    document.getElementById('providerYoutube').checked = s.providers.includes('youtube');
    document.getElementById('providerSoundcloud').checked = s.providers.includes('soundcloud');
    document.getElementById('settingQueueLimit').value = s.limits.queue;
    document.getElementById('settingUserLimit').value = s.limits.user;
    document.getElementById('settingPlaylistOnly').checked = s.limits.playlistOnly;
    document.getElementById('settingExemptLevel').value = s.limits.exemptUserLevel;
    document.getElementById('settingLimitToMusic').checked = s.youtube.limitToMusic;
    document.getElementById('settingLimitToLiked').checked = s.youtube.limitToLikedVideos;
}

async function saveSettings() {
    const providers = [];
    if (document.getElementById('providerYoutube').checked) providers.push('youtube');
    if (document.getElementById('providerSoundcloud').checked) providers.push('soundcloud');

    const body = {
        enabled: document.getElementById('settingEnabled').checked,
        volume: parseInt(document.getElementById('settingVolume').value),
        searchProvider: document.getElementById('settingSearchProvider').value,
        userLevel: document.getElementById('settingUserLevel').value,
        providers,
        limits: {
            queue: parseInt(document.getElementById('settingQueueLimit').value),
            user: parseInt(document.getElementById('settingUserLimit').value),
            playlistOnly: document.getElementById('settingPlaylistOnly').checked,
            exemptUserLevel: document.getElementById('settingExemptLevel').value,
        },
        youtube: {
            limitToMusic: document.getElementById('settingLimitToMusic').checked,
            limitToLikedVideos: document.getElementById('settingLimitToLiked').checked,
        },
    };

    try {
        const resp = await api('PUT', '/song_requests', body);
        state.settings = resp.settings;
        state.volume = resp.settings.volume;
        updateVolumeUI();
        if (ytPlayer && state.playerReady) ytPlayer.setVolume(state.isMuted ? 0 : state.volume);
        document.getElementById('enableToggle').checked = resp.settings.enabled;
        showToast('Settings saved', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ── Actions ──────────────────────────────────────────────────────────
async function skipToNext() {
    try {
        const resp = await api('POST', '/song_requests/queue/skip');
        state.currentSong = resp._currentSong;
        updateNowPlaying();
        if (state.currentSong && state.currentSong.track.provider === 'youtube') {
            loadYouTubeVideo(state.currentSong.track.providerId);
        } else if (!state.currentSong) {
            stopPlayer();
        }
        await refreshQueue();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function playPrevious() {
    if (state.history.length === 0) return;
    const prev = state.history.pop();
    // Push current to queue front
    if (state.currentSong) {
        state.queue.unshift(state.currentSong);
    }
    state.currentSong = prev;
    updateNowPlaying();
    renderQueue();
    if (prev.track.provider === 'youtube') {
        loadYouTubeVideo(prev.track.providerId);
    }
}

async function playItem(itemId) {
    try {
        if (state.currentSong) state.history.push(state.currentSong);
        const resp = await api('POST', `/song_requests/queue/${itemId}/play`);
        state.currentSong = resp.item;
        updateNowPlaying();
        if (resp.item.track.provider === 'youtube') {
            loadYouTubeVideo(resp.item.track.providerId);
        }
        await refreshQueue();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function promoteItem(itemId) {
    try {
        await api('POST', `/song_requests/queue/${itemId}/promote`);
        await refreshQueue();
        showToast('Song promoted to top of queue', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function deleteQueueItem(itemId) {
    try {
        await api('DELETE', `/song_requests/queue/${itemId}`);
        state.queue = state.queue.filter(s => s._id !== itemId);
        renderQueue();
        showToast('Song removed from queue');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function deletePlaylistItem(itemId) {
    try {
        await api('DELETE', `/song_requests/playlist/${itemId}`);
        state.playlist = state.playlist.filter(s => s._id !== itemId);
        state.playlistTotal--;
        renderPlaylist();
        showToast('Song removed from playlist');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function queueFromPlaylist(playlistItemId) {
    try {
        const resp = await api('POST', '/song_requests/queue', { q: playlistItemId, fromPlaylist: true });
        showToast(`"${resp.item.track.title}" added to queue`, 'success');
        await refreshQueue();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function refreshQueue() {
    try {
        const resp = await api('GET', '/song_requests/queue');
        state.queue = resp.queue;
        if (resp._currentSong && !state.currentSong) {
            state.currentSong = resp._currentSong;
            updateNowPlaying();
            if (state.currentSong.track.provider === 'youtube') {
                loadYouTubeVideo(state.currentSong.track.providerId);
            }
        } else if (resp._currentSong) {
            state.currentSong = resp._currentSong;
            updateNowPlaying();
        }
        renderQueue();
    } catch (err) {
        console.error('Failed to refresh queue:', err);
    }
}

async function loadPlaylistPage(page) {
    const offset = page * 20;
    try {
        const resp = await api('GET', `/song_requests/playlist?offset=${offset}&limit=20`);
        state.playlist = resp.playlist;
        state.playlistTotal = resp._total;
        state.playlistOffset = resp._offset;
        renderPlaylist();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ── Modals ───────────────────────────────────────────────────────────
function showSongModal(title, desc, submitText, onSubmit) {
    const modal = document.getElementById('songModal');
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalDesc').textContent = desc;
    document.getElementById('modalSubmit').textContent = submitText;
    document.getElementById('modalInput').value = '';
    modal.classList.remove('hidden');
    document.getElementById('modalInput').focus();

    const submitHandler = async () => {
        const value = document.getElementById('modalInput').value.trim();
        if (!value) return;
        document.getElementById('modalSubmit').disabled = true;
        document.getElementById('modalSubmit').textContent = 'Loading...';
        try {
            await onSubmit(value);
            modal.classList.add('hidden');
        } catch (err) {
            showToast(err.message, 'error');
        }
        document.getElementById('modalSubmit').disabled = false;
        document.getElementById('modalSubmit').textContent = submitText;
        cleanup();
    };

    const cancelHandler = () => {
        modal.classList.add('hidden');
        cleanup();
    };

    const keyHandler = (e) => { if (e.key === 'Enter') submitHandler(); if (e.key === 'Escape') cancelHandler(); };

    function cleanup() {
        document.getElementById('modalSubmit').replaceWith(document.getElementById('modalSubmit').cloneNode(true));
        document.getElementById('modalCancel').replaceWith(document.getElementById('modalCancel').cloneNode(true));
        document.getElementById('modalInput').removeEventListener('keydown', keyHandler);
        document.querySelector('#songModal .modal-backdrop').removeEventListener('click', cancelHandler);
        // Rebind
        document.getElementById('modalCancel').addEventListener('click', cancelHandler);
    }

    document.getElementById('modalSubmit').addEventListener('click', submitHandler);
    document.getElementById('modalCancel').addEventListener('click', cancelHandler);
    document.getElementById('modalInput').addEventListener('keydown', keyHandler);
    document.querySelector('#songModal .modal-backdrop').addEventListener('click', cancelHandler);
}

function showConfirmModal(title, desc, submitText, onConfirm) {
    const modal = document.getElementById('confirmModal');
    document.getElementById('confirmTitle').textContent = title;
    document.getElementById('confirmDesc').textContent = desc;
    document.getElementById('confirmSubmit').textContent = submitText;
    modal.classList.remove('hidden');

    const submitHandler = async () => {
        await onConfirm();
        modal.classList.add('hidden');
        cleanup();
    };

    const cancelHandler = () => {
        modal.classList.add('hidden');
        cleanup();
    };

    function cleanup() {
        document.getElementById('confirmSubmit').replaceWith(document.getElementById('confirmSubmit').cloneNode(true));
        document.getElementById('confirmCancel').replaceWith(document.getElementById('confirmCancel').cloneNode(true));
        document.querySelector('#confirmModal .modal-backdrop').removeEventListener('click', cancelHandler);
    }

    document.getElementById('confirmSubmit').addEventListener('click', submitHandler);
    document.getElementById('confirmCancel').addEventListener('click', cancelHandler);
    document.querySelector('#confirmModal .modal-backdrop').addEventListener('click', cancelHandler);
}

// ── Toasts ───────────────────────────────────────────────────────────
function showToast(message, type = '') {
    const container = document.getElementById('toasts');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.remove(); }, 4000);
}

// ── WebSocket ────────────────────────────────────────────────────────
let ws = null;
let wsRetryDelay = 1000;

async function connectWebSocket() {
    try {
        const resp = await api('GET', '/me/ws_token');
        const token = resp.token;
        ws = new WebSocket(`${WS_BASE}/ws?token=${token}&channel=${CHANNEL_ID}`);

        ws.onopen = () => {
            wsRetryDelay = 1000;
            console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleWSEvent(msg.event, msg.data);
            } catch (e) {
                console.error('WS message parse error:', e);
            }
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected, reconnecting...');
            setTimeout(connectWebSocket, wsRetryDelay);
            wsRetryDelay = Math.min(wsRetryDelay * 2, 30000);
        };

        ws.onerror = () => {
            ws.close();
        };
    } catch (err) {
        setTimeout(connectWebSocket, wsRetryDelay);
        wsRetryDelay = Math.min(wsRetryDelay * 2, 30000);
    }
}

function handleWSEvent(event, data) {
    switch (event) {
        case 'songRequestQueueAdd':
            if (data.item && !state.queue.find(s => s._id === data.item._id)) {
                state.queue.push(data.item);
                renderQueue();
                showToast(`"${data.item.track.title}" added to queue`, 'success');
            }
            break;
        case 'songRequestQueueRemove':
            if (data.item) {
                state.queue = state.queue.filter(s => s._id !== data.item._id);
                renderQueue();
            }
            break;
        case 'songRequestQueueClear':
            state.queue = [];
            renderQueue();
            break;
        case 'songRequestQueuePromote':
            refreshQueue();
            break;
        case 'songRequestPlay':
            if (data.item) {
                state.currentSong = data.item;
                updateNowPlaying();
                if (data.item.track.provider === 'youtube') {
                    loadYouTubeVideo(data.item.track.providerId);
                }
            }
            refreshQueue();
            break;
        case 'songRequestSkip':
            refreshQueue();
            break;
        case 'songRequestPause':
            if (ytPlayer && state.playerReady) ytPlayer.pauseVideo();
            break;
        case 'songRequestVolume':
            if (data.volume != null) {
                state.volume = data.volume;
                updateVolumeUI();
                if (ytPlayer && state.playerReady) ytPlayer.setVolume(state.isMuted ? 0 : state.volume);
            }
            break;
    }
}

// ── Initialization ───────────────────────────────────────────────────
async function init() {
    // Load settings
    try {
        const resp = await api('GET', '/song_requests');
        state.settings = resp.settings;
        state.volume = resp.settings.volume;
        loadSettingsUI();
        updateVolumeUI();
    } catch (err) {
        console.error('Failed to load settings:', err);
    }

    // Load queue
    await refreshQueue();

    // Load playlist
    await loadPlaylistPage(0);

    // Connect WebSocket
    connectWebSocket();

    // ── Event Bindings ─────────────────────────────────────────────
    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab + 'Tab').classList.add('active');
        });
    });

    // Player controls
    document.getElementById('playPauseBtn').addEventListener('click', () => {
        if (!ytPlayer || !state.playerReady) return;
        if (state.isPlaying) {
            ytPlayer.pauseVideo();
        } else {
            ytPlayer.playVideo();
        }
    });

    document.getElementById('nextBtn').addEventListener('click', () => {
        if (state.currentSong) state.history.push(state.currentSong);
        skipToNext();
    });

    document.getElementById('prevBtn').addEventListener('click', playPrevious);

    document.getElementById('shuffleBtn').addEventListener('click', async () => {
        // Shuffle the queue
        const shuffled = [...state.queue];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        state.queue = shuffled;
        renderQueue();
        await api('PATCH', '/song_requests/queue/order', { order: shuffled.map(s => s._id) });
        showToast('Queue shuffled');
    });

    // Volume
    document.getElementById('volumeSlider').addEventListener('input', (e) => {
        state.volume = parseInt(e.target.value);
        state.isMuted = false;
        updateVolumeUI();
        if (ytPlayer && state.playerReady) ytPlayer.setVolume(state.volume);
        // Save to server
        api('PUT', '/song_requests', { volume: state.volume }).catch(() => { });
    });

    document.getElementById('muteBtn').addEventListener('click', () => {
        state.isMuted = !state.isMuted;
        updateVolumeUI();
        if (ytPlayer && state.playerReady) ytPlayer.setVolume(state.isMuted ? 0 : state.volume);
    });

    // Seek
    document.getElementById('seekBar').addEventListener('input', (e) => {
        if (ytPlayer && state.playerReady) {
            ytPlayer.seekTo(parseFloat(e.target.value), true);
        }
    });

    // Search
    document.getElementById('queueSearch').addEventListener('input', renderQueue);
    document.getElementById('playlistSearch').addEventListener('input', renderPlaylist);

    // Request Song
    document.getElementById('requestSongBtn').addEventListener('click', () => {
        showSongModal('Request a Song', 'Enter a song name or URL', 'Request', async (value) => {
            const resp = await api('POST', '/song_requests/queue', { q: value });
            showToast(`"${resp.item.track.title}" added to queue`, 'success');
            await refreshQueue();
        });
    });

    // Clear Queue
    document.getElementById('clearQueueBtn').addEventListener('click', () => {
        showConfirmModal('Clear Queue', 'Are you sure you want to clear the song request queue?', 'Clear Queue', async () => {
            await api('DELETE', '/song_requests/queue');
            state.queue = [];
            renderQueue();
            showToast('Queue cleared');
        });
    });

    // Add to Playlist
    document.getElementById('addToPlaylistBtn').addEventListener('click', () => {
        showSongModal('Add to Playlist', 'Enter a song name or URL', 'Add', async (value) => {
            const resp = await api('POST', '/song_requests/playlist', { q: value });
            showToast(`"${resp.item.track.title}" added to playlist`, 'success');
            await loadPlaylistPage(Math.floor(state.playlistOffset / 20));
        });
    });

    // Import Playlist
    document.getElementById('importPlaylistBtn').addEventListener('click', () => {
        showSongModal('Import YouTube Playlist', 'Enter a YouTube playlist URL', 'Import', async (value) => {
            await api('POST', '/song_requests/playlist/import', { url: value });
            showToast('Playlist import started. Songs will appear shortly.', 'success');
            setTimeout(() => loadPlaylistPage(0), 3000);
        });
    });

    // Clear Playlist
    document.getElementById('clearPlaylistBtn').addEventListener('click', () => {
        showConfirmModal('Clear Playlist', 'Are you sure you want to clear the entire playlist?', 'Clear Playlist', async () => {
            await api('DELETE', '/song_requests/playlist');
            state.playlist = [];
            state.playlistTotal = 0;
            renderPlaylist();
            showToast('Playlist cleared');
        });
    });

    // Save Settings
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);

    // Enable toggle
    document.getElementById('enableToggle').addEventListener('change', async (e) => {
        try {
            await api('PUT', '/song_requests', { enabled: e.target.checked });
            state.settings.enabled = e.target.checked;
            document.getElementById('settingEnabled').checked = e.target.checked;
            showToast(`Song requests ${e.target.checked ? 'enabled' : 'disabled'}`);
        } catch (err) {
            showToast(err.message, 'error');
            e.target.checked = !e.target.checked;
        }
    });

    // Settings volume slider real-time label
    document.getElementById('settingVolume').addEventListener('input', (e) => {
        document.getElementById('settingVolumeLabel').textContent = e.target.value;
    });

    // Settings button in header
    document.getElementById('settingsBtn').addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        document.querySelector('[data-tab="settings"]').classList.add('active');
        document.getElementById('settingsTab').classList.add('active');
    });
}

// Start
document.addEventListener('DOMContentLoaded', init);
