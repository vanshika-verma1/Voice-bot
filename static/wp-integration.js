/**
 * BharatLogic Voice Agent - WordPress Integration Script
 * This script adds a floating voice assistant widget to the corner of the website.
 */

(function () {
    // 1. Configuration - Replace with your backend URL
    // 1. Configuration - Automatically detect backend if running on same server
    const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.hostname.includes('ngrok-free.app')
        ? window.location.origin
        : 'http://localhost:8000'; // Default fallback

    // 2. Inject Styles
    const styles = `
        #voice-agent-widget {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 15px;
        }
        #voice-agent-chat {
            width: 380px;
            height: 550px;
            background: #0A1628;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            border: 1px solid #2D3F59;
            display: none;
            flex-direction: column;
            overflow: hidden;
            transition: all 0.3s ease;
            transform-origin: bottom right;
            transform: scale(0.8);
            opacity: 0;
        }
        #voice-agent-chat.open {
            display: flex;
            transform: scale(1);
            opacity: 1;
        }
        #voice-agent-header {
            background: #162032;
            padding: 12px 15px;
            border-bottom: 1px solid #2D3F59;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-main {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #voice-agent-log {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
            scrollbar-width: thin;
            scrollbar-color: #2D3F59 transparent;
        }
        .va-message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 16px;
            font-size: 14px;
            line-height: 1.5;
            word-wrap: break-word;
            animation: va-slide-in 0.3s ease;
        }
        @keyframes va-slide-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .va-user {
            align-self: flex-end;
            background: #0066FF;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .va-agent {
            align-self: flex-start;
            background: #1A2940;
            color: #E2E8F0;
            border-bottom-left-radius: 4px;
            border: 1px solid #2D3F59;
        }
        .va-system {
            align-self: center;
            background: rgba(255, 255, 255, 0.05);
            color: #94A3B8;
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 20px;
        }
        #voice-agent-button {
            width: 65px;
            height: 65px;
            border-radius: 50%;
            background: #0066FF;
            box-shadow: 0 10px 25px rgba(0,102,255,0.3);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: none;
            color: white;
        }
        #voice-agent-button:hover {
            transform: scale(1.05);
        }
        #voice-agent-button.active {
            background: #EF4444;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
        }
        .va-controls {
            padding: 12px 15px;
            background: #162032;
            border-top: 1px solid #2D3F59;
            display: flex;
            gap: 10px;
        }
        .va-btn {
            flex: 1;
            padding: 10px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.2s;
        }
        #va-restart { background: #0066FF; color: white; }
        #va-end { background: rgba(239, 68, 68, 0.1); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .va-btn:hover:not(:disabled) { filter: brightness(1.1); }
        .va-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        @keyframes va-pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .va-recording #voice-agent-button svg {
            animation: va-pulse 1.2s infinite ease-in-out;
        }
        .highlight-target {
            outline: 3px solid #0066FF;
            outline-offset: 4px;
            transition: all 0.3s;
            position: relative;
            z-index: 1000;
        }
    `;

    const styleSheet = document.createElement("style");
    styleSheet.innerText = styles;
    document.head.appendChild(styleSheet);

    // 3. Create UI
    const widget = document.createElement('div');
    widget.id = 'voice-agent-widget';
    widget.innerHTML = `
        <div id="voice-agent-chat">
            <div id="voice-agent-header">
                <div class="header-main">
                    <strong>Aura AI Assistant</strong>
                    <span id="va-status-dot" style="width: 8px; height: 8px; border-radius: 50%; background: #94A3B8;"></span>
                </div>
                <button id="va-close-chat" style="background:none; border:none; color:#94A3B8; cursor:pointer; font-size:20px;">&times;</button>
            </div>
            <div id="voice-agent-log">
                <!-- Messages will appear here -->
            </div>
            <div class="va-controls">
                <button id="va-restart" class="va-btn">Restart</button>
                <button id="va-end" class="va-btn">End session</button>
            </div>
        </div>
        <button id="voice-agent-button" title="Talk to Aura">
            <svg viewBox="0 0 24 24" width="30" height="30" fill="currentColor">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
            </svg>
        </button>
        <div id="voice-agent-audio-container" style="display: none;"></div>
    `;
    document.body.appendChild(widget);

    const btn = document.getElementById('voice-agent-button');
    const chat = document.getElementById('voice-agent-chat');
    const log = document.getElementById('voice-agent-log');
    const statusDot = document.getElementById('va-status-dot');
    const btnRestart = document.getElementById('va-restart');
    const btnEnd = document.getElementById('va-end');
    const btnClose = document.getElementById('va-close-chat');

    function bindLinks() {
        document.querySelectorAll('a').forEach(link => {
            const href = link.getAttribute('href');
            if (href && !href.startsWith('http') && !href.startsWith('//') && !href.startsWith('mailto:') && !href.startsWith('tel:') && !href.startsWith('#')) {
                link.onclick = (e) => {
                    e.preventDefault();
                    websiteActions.navigate(href);
                };
            }
        });
    }

    bindLinks();

    window.onpopstate = (e) => {
        if (e.state && e.state.path) {
            websiteActions.navigate(e.state.path);
        }
    };

    function addMessage(role, text, isSystem = false) {
        const div = document.createElement('div');
        div.className = isSystem ? 'va-system' : `va-message ${role === 'user' ? 'va-user' : 'va-agent'}`;
        div.textContent = text;
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
        return div;
    }

    // 4. LiveKit Logic
    let currentRoom = null;
    let sessionEnded = false;
    const activeTranscriptions = new Map(); // key: pId, value: element

    const websiteActions = {
        navigate: async (url) => {
            if (url.startsWith('#')) {
                const el = document.querySelector(url);
                if (el) el.scrollIntoView({ behavior: 'smooth' });
                return;
            }
            try {
                const response = await fetch(url);
                const html = await response.text();
                const parser = new DOMParser();
                const newDoc = parser.parseFromString(html, 'text/html');
                document.title = newDoc.title;

                const widget = document.getElementById('voice-agent-widget');
                Array.from(document.body.childNodes).forEach(node => {
                    if (node !== widget && node.nodeName !== 'SCRIPT' && node.nodeName !== 'STYLE') {
                        document.body.removeChild(node);
                    }
                });

                const newContent = document.createElement('div');
                newContent.id = 'spa-content-wrapper';
                newContent.innerHTML = newDoc.body.innerHTML;
                document.body.insertBefore(newContent, widget);

                window.history.pushState({ path: url }, '', url);
                bindLinks();
            } catch (err) {
                window.location.href = url;
            }
        },
        scroll: (selector) => {
            const el = document.querySelector(selector);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        },
        highlight: (selector) => {
            const el = document.querySelector(selector);
            if (el) {
                document.querySelectorAll('.highlight-target').forEach(e => e.classList.remove('highlight-target'));
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                el.classList.add('highlight-target');
                setTimeout(() => el.classList.remove('highlight-target'), 5000);
            }
        },
        click: (selector) => {
            const el = document.querySelector(selector);
            if (el) {
                el.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    el.style.transform = '';
                    el.click();
                }, 100);
            }
        }
    };

    window.auraWebsiteActions = websiteActions;

    async function startAgent() {
        if (currentRoom) return;

        try {
            const { Room, RoomEvent } = await import('https://cdn.jsdelivr.net/npm/livekit-client@2.5.5/+esm');

            chat.classList.add('open');
            statusDot.style.background = '#10B981';
            localStorage.setItem('aura_agent_active', 'true');
            sessionEnded = false;

            const response = await fetch(`${BACKEND_URL}/getToken`);
            const data = await response.json();

            const room = new Room({ adaptiveStream: true });
            currentRoom = room;

            room.on(RoomEvent.TranscriptionReceived, (segments, participant) => {
                const isLocal = participant && participant.identity === room.localParticipant.identity;
                const role = isLocal ? 'user' : 'agent';
                const pId = participant.identity;

                segments.forEach(segment => {
                    if (!segment.text) return;

                    let msgEl = activeTranscriptions.get(pId);
                    if (!msgEl) {
                        msgEl = addMessage(role, "");
                        activeTranscriptions.set(pId, msgEl);
                    }

                    msgEl.textContent = segment.text;
                    log.scrollTop = log.scrollHeight;

                    if (segment.final) {
                        activeTranscriptions.delete(pId);
                    }
                });
            });

            room.on(RoomEvent.Disconnected, () => {
                cleanupUI();
            });

            const audioContainer = document.getElementById('voice-agent-audio-container');
            room.on(RoomEvent.TrackSubscribed, (track) => {
                if (track.kind === 'audio') {
                    const audioEl = track.attach();
                    audioContainer.appendChild(audioEl);
                }
            });

            await room.connect(data.url, data.token);
            await room.localParticipant.setMicrophoneEnabled(true);

            room.on(RoomEvent.DataReceived, (payload, participant) => {
                const decoder = new TextDecoder();
                const strData = decoder.decode(payload);
                try {
                    const data = JSON.parse(strData);
                    if (data.type === 'WEBSITE_CONTROL') {
                        const { action, target } = data;
                        console.log(`Command received: ${action} -> ${target}`);
                        if (websiteActions[action]) {
                            websiteActions[action](target);
                        }
                    }
                } catch (e) {
                    console.error("Failed to parse data message:", e);
                }
            });


            btn.classList.add('active');
            widget.classList.add('va-recording');
            btnEnd.disabled = false;
            btnRestart.disabled = false;

        } catch (err) {
            console.error('LiveKit Error:', err);
            statusDot.style.background = '#EF4444';
            cleanupUI();
        }
    }

    function cleanupUI() {
        btn.classList.remove('active');
        widget.classList.remove('va-recording');
        statusDot.style.background = '#94A3B8';
        currentRoom = null;
        activeTranscriptions.clear();
        btnEnd.disabled = true;
        btnRestart.disabled = false;
    }

    function stopAgent() {
        if (currentRoom) {
            currentRoom.disconnect();
            localStorage.removeItem('aura_agent_active');
        }
    }

    function endSessionUI() {
        stopAgent();
        sessionEnded = true;
        addMessage("", "Conversation ended", true);
        localStorage.removeItem('aura_agent_active');
    }

    function restartSession() {
        stopAgent();
        log.innerHTML = '<div class="va-message va-agent">Hi! I\'m Aura. How can I help you today?</div>';
        setTimeout(startAgent, 500);
    }

    btn.onclick = async () => {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioContext();
            if (ctx.state === 'suspended') await ctx.resume();
        } catch (e) { }

        if (currentRoom) {
            stopAgent();
        } else {
            startAgent();
        }
    };

    btnRestart.onclick = restartSession;
    btnEnd.onclick = endSessionUI;
    btnClose.onclick = () => chat.classList.remove('open');

    window.addEventListener('load', () => {
        if (localStorage.getItem('aura_agent_active') === 'true') {
            setTimeout(startAgent, 1000);
        }
    });

    window.addEventListener('beforeunload', () => {
        if (currentRoom) currentRoom.disconnect();
    });

})();
