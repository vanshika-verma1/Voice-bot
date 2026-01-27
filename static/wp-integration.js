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
            width: 350px;
            height: 500px;
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
            padding: 15px;
            border-bottom: 1px solid #2D3F59;
            color: white;
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
            gap: 10px;
            scrollbar-width: thin;
            scrollbar-color: #2D3F59 transparent;
        }
        .va-message {
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.4;
        }
        .va-user {
            align-self: flex-end;
            background: #0066FF;
            color: white;
            border-bottom-right-radius: 2px;
        }
        .va-agent {
            align-self: flex-start;
            background: #1A2940;
            color: #E2E8F0;
            border-bottom-left-radius: 2px;
            border: 1px solid #2D3F59;
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
        @keyframes va-pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .va-recording #voice-agent-button svg {
            animation: va-pulse 1.2s infinite ease-in-out;
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
                <strong>Aura AI Assistant</strong>
                <span id="va-status-dot" style="width: 8px; height: 8px; border-radius: 50%; background: #94A3B8;"></span>
            </div>
            <div id="voice-agent-log">
                <div class="va-message va-agent">Hi! I'm Aura. How can I help you today?</div>
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

    function bindLinks() {
        document.querySelectorAll('a').forEach(link => {
            // Only bind internal links
            const href = link.getAttribute('href');
            if (href && !href.startsWith('http') && !href.startsWith('//') && !href.startsWith('mailto:') && !href.startsWith('tel:')) {
                link.onclick = (e) => {
                    e.preventDefault();
                    websiteActions.navigate(href);
                };
            }
        });
    }

    // Initialize links
    bindLinks();

    // Handle browser back/forward
    window.onpopstate = (e) => {
        if (e.state && e.state.path) {
            websiteActions.navigate(e.state.path);
        }
    };

    function addMessage(role, text) {
        const div = document.createElement('div');
        div.className = `va-message ${role === 'user' ? 'va-user' : 'va-agent'}`;
        div.textContent = text;
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
    }

    // 4. LiveKit Logic
    let currentRoom = null;

    // Website Control Actions
    const websiteActions = {
        navigate: async (url) => {
            console.log('SPA Navigating to:', url);

            // If it's just a hash/anchor, scroll to it
            if (url.startsWith('#')) {
                const el = document.querySelector(url);
                if (el) el.scrollIntoView({ behavior: 'smooth' });
                return;
            }

            try {
                // Fetch the new page content
                const response = await fetch(url);
                const html = await response.text();
                const parser = new DOMParser();
                const newDoc = parser.parseFromString(html, 'text/html');

                // 1. Update the document title
                document.title = newDoc.title;

                // 2. Update the main content safely
                const newBody = newDoc.body;
                const widget = document.getElementById('voice-agent-widget');

                // Remove everything EXCEPT the widget and essential tags
                Array.from(document.body.childNodes).forEach(node => {
                    if (node !== widget && node.nodeName !== 'SCRIPT' && node.nodeName !== 'STYLE') {
                        document.body.removeChild(node);
                    }
                });

                // Add new content (wrapped in a div to avoid fragmentation if needed)
                const newContent = document.createElement('div');
                newContent.id = 'spa-content-wrapper';
                newContent.innerHTML = newBody.innerHTML;
                document.body.insertBefore(newContent, widget);

                // 3. Update URL
                window.history.pushState({ path: url }, '', url);

                // 4. Re-bind links to use SPA navigation too
                bindLinks();

                console.log('SPA Navigation successful');
            } catch (err) {
                console.error('SPA Navigation failed, falling back to reload:', err);
                window.location.href = url;
            }
        },
        scroll: (selector) => {
            console.log('Scrolling to:', selector);
            const el = document.querySelector(selector);
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        },
        highlight: (selector) => {
            console.log('Highlighting:', selector);
            const el = document.querySelector(selector);
            if (el) {
                // Remove previous highlights
                document.querySelectorAll('.highlight-target').forEach(e => e.classList.remove('highlight-target'));

                // Scroll to it first
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });

                // Add highlight class
                el.classList.add('highlight-target');

                // Auto-remove highlight after 5 seconds
                setTimeout(() => {
                    el.classList.remove('highlight-target');
                }, 5000);
            }
        },
        click: (selector) => {
            console.log('Clicking:', selector);
            const el = document.querySelector(selector);
            if (el) {
                // Visual feedback before click
                el.style.transform = 'scale(0.95)';
                el.style.transition = 'transform 0.1s';

                setTimeout(() => {
                    el.style.transform = '';
                    el.click();
                }, 100);
            }
        }
    };

    // Expose actions to window for testing/integration purposes
    window.auraWebsiteActions = websiteActions;

    async function startAgent() {
        try {
            const { Room, RoomEvent } = await import('https://cdn.jsdelivr.net/npm/livekit-client@2.5.5/+esm');

            chat.classList.add('open');
            statusDot.style.background = '#10B981';
            localStorage.setItem('aura_agent_active', 'true');

            const response = await fetch(`${BACKEND_URL}/getToken`);
            const data = await response.json();

            const room = new Room();
            currentRoom = room;

            // Handle room events
            room.on(RoomEvent.TranscriptionReceived, (transcriptions, participant) => {
                const text = transcriptions.map(t => t.text).join(' ');
                if (!text) return;

                // Improved participant check
                const isLocal = participant && participant.identity === room.localParticipant.identity;
                addMessage(isLocal ? 'user' : 'agent', text);
            });

            room.on(RoomEvent.DataReceived, (payload, participant) => {
                const decoder = new TextDecoder();
                const str = decoder.decode(payload);
                try {
                    const data = JSON.parse(str);
                    if (data.type === 'WEBSITE_CONTROL') {
                        const { action, target } = data;
                        if (websiteActions[action]) {
                            websiteActions[action](target);
                        }
                    }
                } catch (e) {
                    console.error('Error parsing data channel message:', e);
                }
            });

            room.on(RoomEvent.Disconnected, () => {
                btn.classList.remove('active');
                widget.classList.remove('va-recording');
                statusDot.style.background = '#94A3B8';
                currentRoom = null;
                localStorage.removeItem('aura_agent_active');
            });

            // Handle incoming tracks (AUDIO) - BEFORE CONNECT
            const audioContainer = document.getElementById('voice-agent-audio-container');
            room.on(RoomEvent.TrackSubscribed, (track) => {
                if (track.kind === 'audio') {
                    console.log('Audio track subscribed, attaching to persistent container...');
                    const audioEl = track.attach();
                    audioContainer.appendChild(audioEl);
                }
            });

            room.on(RoomEvent.TrackUnsubscribed, (track) => {
                track.detach().forEach(el => el.remove());
            });

            await room.connect(data.url, data.token);
            await room.localParticipant.setMicrophoneEnabled(true);

            btn.classList.add('active');
            widget.classList.add('va-recording');

        } catch (err) {
            console.error('LiveKit Error:', err);
            statusDot.style.background = '#EF4444';
            btn.classList.remove('active');
            widget.classList.remove('va-recording');
            localStorage.removeItem('aura_agent_active');
        }
    }

    function stopAgent() {
        if (currentRoom) {
            currentRoom.disconnect();
            chat.classList.remove('open');
            localStorage.removeItem('aura_agent_active');
        }
    }

    btn.onclick = async () => {
        // Resume AudioContext just in case browser blocked it
        try {
            const { AudioContext } = window.AudioContext ? window : window.webkitAudioContext;
            const ctx = new AudioContext();
            if (ctx.state === 'suspended') await ctx.resume();
        } catch (e) { }

        if (currentRoom) {
            stopAgent();
        } else {
            startAgent();
        }
    };

    // Auto-reconnect if it was active before refresh
    window.addEventListener('load', () => {
        if (localStorage.getItem('aura_agent_active') === 'true') {
            console.log('Aura: Restoring persistent session...');
            setTimeout(startAgent, 1000); // Small delay to let page settle
        }
    });

})();
