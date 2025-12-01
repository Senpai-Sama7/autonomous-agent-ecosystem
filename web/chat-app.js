/**
 * ASTRO Chat Application
 * Consumer-grade AI assistant interface
 * 
 * Features:
 * - Intuitive chat interface like Claude/ChatGPT
 * - Conversation history management
 * - File attachments
 * - Code highlighting and artifacts
 * - Real-time streaming responses
 * - Accessibility support
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
  API_BASE: window.location.origin,
  WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ALLOWED_FILE_TYPES: ['.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.css', '.pdf'],
  AUTO_SCROLL: true,
  TYPING_INDICATOR_DELAY: 500,
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

class Store {
  constructor(initialState = {}) {
    this.state = {
      currentSessionId: null,
      sessions: [],
      messages: [],
      isLoading: false,
      isTyping: false,
      attachedFiles: [],
      settings: {
        theme: 'system',
        fontSize: 'medium',
        soundEnabled: false,
        typingIndicator: true,
        autoScroll: true,
      },
      ...initialState,
    };
    this.listeners = new Set();
    this.loadFromStorage();
  }

  getState() {
    return this.state;
  }

  setState(updates) {
    const prevState = { ...this.state };
    this.state = { ...this.state, ...updates };
    this.notify(prevState);
    this.saveToStorage();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(prevState) {
    this.listeners.forEach(listener => listener(this.state, prevState));
  }

  loadFromStorage() {
    try {
      const saved = localStorage.getItem('astro_chat_state');
      if (saved) {
        const parsed = JSON.parse(saved);
        this.state.sessions = parsed.sessions || [];
        this.state.settings = { ...this.state.settings, ...parsed.settings };
      }
    } catch (e) {
      console.warn('Failed to load state from storage', e);
    }
  }

  saveToStorage() {
    try {
      const toSave = {
        sessions: this.state.sessions,
        settings: this.state.settings,
      };
      localStorage.setItem('astro_chat_state', JSON.stringify(toSave));
    } catch (e) {
      console.warn('Failed to save state to storage', e);
    }
  }
}

const store = new Store();

// ============================================================================
// API CLIENT
// ============================================================================

class APIClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  async request(endpoint, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error('Request timed out');
      }
      throw error;
    }
  }

  async sendMessage(sessionId, message, files = []) {
    return this.request('/api/chat/message', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        message,
        files: files.map(f => ({ name: f.name, type: f.type, size: f.size })),
      }),
    });
  }

  async getHistory(sessionId) {
    return this.request(`/api/chat/history/${sessionId}`);
  }

  async getSessions() {
    return this.request('/api/chat/sessions');
  }

  async getSystemStatus() {
    return this.request('/api/system/status');
  }

  async startSystem() {
    return this.request('/api/system/start', { method: 'POST' });
  }
}

const api = new APIClient(CONFIG.API_BASE);

// ============================================================================
// WEBSOCKET CLIENT
// ============================================================================

class WebSocketClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.handlers = new Map();
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (e) {
          console.warn('Failed to parse WebSocket message', e);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket', error);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    setTimeout(() => this.connect(), delay);
  }

  handleMessage(data) {
    const { type, payload } = data;
    this.emit(type, payload);
    
    // Handle specific message types
    if (type === 'chat_response') {
      this.emit('response', payload);
    } else if (type === 'typing_start') {
      store.setState({ isTyping: true });
    } else if (type === 'typing_end') {
      store.setState({ isTyping: false });
    }
  }

  send(type, payload) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    }
  }

  on(event, handler) {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event).add(handler);
    return () => this.handlers.get(event).delete(handler);
  }

  emit(event, data) {
    if (this.handlers.has(event)) {
      this.handlers.get(event).forEach(handler => handler(data));
    }
  }
}

const ws = new WebSocketClient(CONFIG.WS_URL);

// ============================================================================
// UI COMPONENTS
// ============================================================================

const UI = {
  // Generate unique IDs
  generateId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },

  // Format timestamp
  formatTime(date) {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(date);
  },

  // Format date for grouping
  formatDate(date) {
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  },

  // Escape HTML
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  // Parse markdown-like formatting
  parseContent(text) {
    // Escape HTML first
    let html = this.escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'plaintext';
      return `
        <div class="artifact">
          <div class="artifact-header">
            <span class="artifact-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
              </svg>
              ${language}
            </span>
            <div class="artifact-actions">
              <button class="copy-code-btn" onclick="UI.copyCode(this)">Copy</button>
            </div>
          </div>
          <div class="artifact-body">
            <pre><code class="language-${language}">${code.trim()}</code></pre>
          </div>
        </div>
      `;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Line breaks to paragraphs
    html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');
    html = html.replace(/\n/g, '<br>');

    return html;
  },

  // Copy code to clipboard
  async copyCode(button) {
    const codeBlock = button.closest('.artifact').querySelector('code');
    const text = codeBlock.textContent;

    try {
      await navigator.clipboard.writeText(text);
      const originalText = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = originalText;
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  },

  // Render a single message
  renderMessage(message) {
    const isUser = message.role === 'user';
    const time = message.timestamp ? this.formatTime(new Date(message.timestamp)) : '';

    return `
      <div class="message message--${message.role}" data-id="${message.id}">
        <div class="message-avatar">
          ${isUser ? '👤' : '✦'}
        </div>
        <div class="message-content">
          <div class="message-header">
            <span class="message-author">${isUser ? 'You' : 'ASTRO'}</span>
            <span class="message-time">${time}</span>
          </div>
          <div class="message-body">
            ${this.parseContent(message.content)}
          </div>
          <div class="message-actions">
            <button class="message-action-btn" onclick="UI.copyMessage('${message.id}')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"></path>
              </svg>
              Copy
            </button>
            ${!isUser ? `
              <button class="message-action-btn" onclick="UI.regenerateMessage('${message.id}')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M23 4v6h-6M1 20v-6h6"></path>
                  <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"></path>
                </svg>
                Regenerate
              </button>
            ` : ''}
          </div>
        </div>
      </div>
    `;
  },

  // Render typing indicator
  renderTypingIndicator() {
    return `
      <div class="message message--assistant typing-message">
        <div class="message-avatar">✦</div>
        <div class="message-content">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    `;
  },

  // Render conversation list item
  renderConversationItem(session) {
    const isActive = session.id === store.getState().currentSessionId;
    return `
      <li>
        <button class="conv-btn ${isActive ? 'active' : ''}" data-session-id="${session.id}">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"></path>
          </svg>
          <span class="conv-title">${this.escapeHtml(session.title || 'New conversation')}</span>
        </button>
      </li>
    `;
  },

  // Copy message content
  async copyMessage(messageId) {
    const { messages } = store.getState();
    const message = messages.find(m => m.id === messageId);
    if (message) {
      try {
        await navigator.clipboard.writeText(message.content);
        // Show toast notification
        this.showToast('Copied to clipboard');
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    }
  },

  // Regenerate message (placeholder)
  regenerateMessage(messageId) {
    console.log('Regenerate message:', messageId);
    // TODO: Implement regeneration
    this.showToast('Regeneration coming soon');
  },

  // Show toast notification
  showToast(message, duration = 3000) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 100px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--text-primary);
      color: var(--bg-primary);
      padding: 12px 24px;
      border-radius: var(--radius-md);
      font-size: 0.875rem;
      z-index: 1001;
      animation: fadeIn 0.2s ease;
    `;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = 'fadeOut 0.2s ease';
      setTimeout(() => toast.remove(), 200);
    }, duration);
  },
};

// Make UI globally available for onclick handlers
window.UI = UI;

// ============================================================================
// APPLICATION CONTROLLER
// ============================================================================

class ChatApp {
  constructor() {
    this.elements = {};
    this.init();
  }

  init() {
    this.cacheElements();
    this.bindEvents();
    this.applySettings();
    this.loadSessions();
    ws.connect();

    // Subscribe to state changes
    store.subscribe((state, prevState) => this.onStateChange(state, prevState));
  }

  cacheElements() {
    this.elements = {
      sidebar: document.getElementById('sidebar'),
      welcomeScreen: document.getElementById('welcome-screen'),
      chatContainer: document.getElementById('chat-container'),
      messagesWrapper: document.getElementById('messages-wrapper'),
      messageInput: document.getElementById('message-input'),
      sendBtn: document.getElementById('send-btn'),
      newChatBtn: document.getElementById('new-chat-btn'),
      attachBtn: document.getElementById('attach-btn'),
      fileInput: document.getElementById('file-input'),
      attachedFiles: document.getElementById('attached-files'),
      searchInput: document.getElementById('search-input'),
      capabilitiesBtn: document.getElementById('capabilities-btn'),
      settingsBtn: document.getElementById('settings-btn'),
      capabilitiesModal: document.getElementById('capabilities-modal'),
      settingsModal: document.getElementById('settings-modal'),
      todayConversations: document.getElementById('today-conversations'),
      weekConversations: document.getElementById('week-conversations'),
      olderConversations: document.getElementById('older-conversations'),
    };
  }

  bindEvents() {
    // Message input
    this.elements.messageInput.addEventListener('input', () => this.onInputChange());
    this.elements.messageInput.addEventListener('keydown', (e) => this.onInputKeydown(e));
    this.elements.sendBtn.addEventListener('click', () => this.sendMessage());

    // New chat
    this.elements.newChatBtn.addEventListener('click', () => this.startNewChat());

    // File attachment
    this.elements.attachBtn.addEventListener('click', () => this.elements.fileInput.click());
    this.elements.fileInput.addEventListener('change', (e) => this.onFilesSelected(e));

    // Capability cards and prompt chips
    document.querySelectorAll('[data-prompt]').forEach(el => {
      el.addEventListener('click', () => {
        this.elements.messageInput.value = el.dataset.prompt;
        this.elements.messageInput.focus();
        this.onInputChange();
      });
    });

    // Modals
    this.elements.capabilitiesBtn.addEventListener('click', () => this.openModal('capabilities'));
    this.elements.settingsBtn.addEventListener('click', () => this.openModal('settings'));
    document.querySelectorAll('[data-close-modal]').forEach(btn => {
      btn.addEventListener('click', () => this.closeModals());
    });
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) this.closeModals();
      });
    });

    // Settings
    document.getElementById('theme-select')?.addEventListener('change', (e) => {
      this.updateSetting('theme', e.target.value);
    });
    document.getElementById('clear-history-btn')?.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all conversation history?')) {
        this.clearHistory();
      }
    });

    // Conversation list clicks
    document.addEventListener('click', (e) => {
      const convBtn = e.target.closest('[data-session-id]');
      if (convBtn) {
        this.loadSession(convBtn.dataset.sessionId);
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Escape to close modals
      if (e.key === 'Escape') this.closeModals();
      // Cmd/Ctrl + N for new chat
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault();
        this.startNewChat();
      }
    });

    // WebSocket events
    ws.on('response', (data) => this.onResponse(data));
    ws.on('connected', () => this.onWsConnected());
    ws.on('disconnected', () => this.onWsDisconnected());
  }

  onInputChange() {
    const value = this.elements.messageInput.value.trim();
    this.elements.sendBtn.disabled = value.length === 0;

    // Auto-resize textarea
    const textarea = this.elements.messageInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  }

  onInputKeydown(e) {
    // Enter to send (without shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!this.elements.sendBtn.disabled) {
        this.sendMessage();
      }
    }
  }

  async sendMessage() {
    const content = this.elements.messageInput.value.trim();
    if (!content) return;

    const { currentSessionId, attachedFiles } = store.getState();
    const sessionId = currentSessionId || this.createNewSession();

    // Create user message
    const userMessage = {
      id: UI.generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    // Update state
    store.setState({
      currentSessionId: sessionId,
      messages: [...store.getState().messages, userMessage],
      isLoading: true,
      attachedFiles: [],
    });

    // Clear input
    this.elements.messageInput.value = '';
    this.elements.messageInput.style.height = 'auto';
    this.elements.sendBtn.disabled = true;

    // Show chat container
    this.showChatView();

    // Render user message
    this.renderMessages();

    // Show typing indicator
    store.setState({ isTyping: true });

    try {
      // Send to API
      const response = await api.sendMessage(sessionId, content, attachedFiles);

      // Create assistant message
      const assistantMessage = {
        id: UI.generateId(),
        role: 'assistant',
        content: response.message,
        timestamp: new Date().toISOString(),
      };

      store.setState({
        messages: [...store.getState().messages, assistantMessage],
        isLoading: false,
        isTyping: false,
      });

      // Update session title if first message
      this.updateSessionTitle(sessionId, content);

      // Render messages
      this.renderMessages();

    } catch (error) {
      console.error('Failed to send message:', error);
      store.setState({ isLoading: false, isTyping: false });
      UI.showToast('Failed to send message. Please try again.');
    }
  }

  createNewSession() {
    const sessionId = `session_${Date.now()}`;
    const session = {
      id: sessionId,
      title: 'New conversation',
      createdAt: new Date().toISOString(),
    };

    store.setState({
      sessions: [session, ...store.getState().sessions],
      currentSessionId: sessionId,
    });

    this.renderConversationList();
    return sessionId;
  }

  updateSessionTitle(sessionId, firstMessage) {
    const { sessions } = store.getState();
    const session = sessions.find(s => s.id === sessionId);
    if (session && session.title === 'New conversation') {
      // Use first ~50 chars of message as title
      session.title = firstMessage.slice(0, 50) + (firstMessage.length > 50 ? '...' : '');
      store.setState({ sessions: [...sessions] });
      this.renderConversationList();
    }
  }

  startNewChat() {
    store.setState({
      currentSessionId: null,
      messages: [],
    });
    this.showWelcomeView();
    this.elements.messageInput.focus();
    this.renderConversationList();
  }

  async loadSession(sessionId) {
    store.setState({
      currentSessionId: sessionId,
      isLoading: true,
    });

    try {
      const history = await api.getHistory(sessionId);
      store.setState({
        messages: history || [],
        isLoading: false,
      });
      this.showChatView();
      this.renderMessages();
      this.renderConversationList();
    } catch (error) {
      console.error('Failed to load session:', error);
      store.setState({ isLoading: false });
      UI.showToast('Failed to load conversation');
    }
  }

  async loadSessions() {
    try {
      const sessions = await api.getSessions();
      if (sessions && sessions.length > 0) {
        store.setState({ sessions });
        this.renderConversationList();
      }
    } catch (error) {
      // Sessions endpoint might not exist yet, that's okay
      console.log('Could not load sessions:', error.message);
    }
  }

  showWelcomeView() {
    this.elements.welcomeScreen.classList.remove('hidden');
    this.elements.chatContainer.classList.add('hidden');
  }

  showChatView() {
    this.elements.welcomeScreen.classList.add('hidden');
    this.elements.chatContainer.classList.remove('hidden');
  }

  renderMessages() {
    const { messages, isTyping } = store.getState();
    
    let html = messages.map(msg => UI.renderMessage(msg)).join('');
    
    if (isTyping) {
      html += UI.renderTypingIndicator();
    }

    this.elements.messagesWrapper.innerHTML = html;

    // Auto-scroll to bottom
    if (store.getState().settings.autoScroll) {
      this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
    }
  }

  renderConversationList() {
    const { sessions, currentSessionId } = store.getState();
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

    const todaySessions = [];
    const weekSessions = [];
    const olderSessions = [];

    sessions.forEach(session => {
      const date = new Date(session.createdAt);
      if (date >= today) {
        todaySessions.push(session);
      } else if (date >= weekAgo) {
        weekSessions.push(session);
      } else {
        olderSessions.push(session);
      }
    });

    this.elements.todayConversations.innerHTML = todaySessions.map(s => UI.renderConversationItem(s)).join('');
    this.elements.weekConversations.innerHTML = weekSessions.map(s => UI.renderConversationItem(s)).join('');
    this.elements.olderConversations.innerHTML = olderSessions.map(s => UI.renderConversationItem(s)).join('');
  }

  onFilesSelected(e) {
    const files = Array.from(e.target.files);
    const validFiles = files.filter(file => {
      if (file.size > CONFIG.MAX_FILE_SIZE) {
        UI.showToast(`File ${file.name} is too large (max 10MB)`);
        return false;
      }
      return true;
    });

    store.setState({
      attachedFiles: [...store.getState().attachedFiles, ...validFiles],
    });

    this.renderAttachedFiles();
    e.target.value = ''; // Reset input
  }

  renderAttachedFiles() {
    const { attachedFiles } = store.getState();
    
    this.elements.attachedFiles.innerHTML = attachedFiles.map((file, index) => `
      <div class="attached-file">
        <span>${UI.escapeHtml(file.name)}</span>
        <span class="remove-file" onclick="app.removeFile(${index})">&times;</span>
      </div>
    `).join('');
  }

  removeFile(index) {
    const { attachedFiles } = store.getState();
    attachedFiles.splice(index, 1);
    store.setState({ attachedFiles: [...attachedFiles] });
    this.renderAttachedFiles();
  }

  openModal(type) {
    if (type === 'capabilities') {
      this.elements.capabilitiesModal.classList.remove('hidden');
    } else if (type === 'settings') {
      this.elements.settingsModal.classList.remove('hidden');
    }
  }

  closeModals() {
    this.elements.capabilitiesModal.classList.add('hidden');
    this.elements.settingsModal.classList.add('hidden');
  }

  updateSetting(key, value) {
    const { settings } = store.getState();
    store.setState({
      settings: { ...settings, [key]: value },
    });
    this.applySettings();
  }

  applySettings() {
    const { settings } = store.getState();
    
    // Apply theme
    if (settings.theme === 'system') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', settings.theme);
    }

    // Apply font size
    document.documentElement.style.fontSize = {
      small: '14px',
      medium: '16px',
      large: '18px',
    }[settings.fontSize] || '16px';
  }

  clearHistory() {
    store.setState({
      sessions: [],
      messages: [],
      currentSessionId: null,
    });
    localStorage.removeItem('astro_chat_state');
    this.showWelcomeView();
    this.renderConversationList();
    this.closeModals();
    UI.showToast('Conversation history cleared');
  }

  onStateChange(state, prevState) {
    // Handle state changes that need UI updates
    if (state.isTyping !== prevState.isTyping) {
      this.renderMessages();
    }
  }

  onResponse(data) {
    // Handle streaming response from WebSocket
    const { messages } = store.getState();
    const lastMessage = messages[messages.length - 1];
    
    if (lastMessage && lastMessage.role === 'assistant') {
      lastMessage.content += data.content;
      store.setState({ messages: [...messages] });
      this.renderMessages();
    }
  }

  onWsConnected() {
    console.log('WebSocket connected');
  }

  onWsDisconnected() {
    console.log('WebSocket disconnected');
  }
}

// Make app globally available
window.app = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new ChatApp();
});
