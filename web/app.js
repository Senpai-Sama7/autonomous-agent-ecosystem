/**
 * ASTRO Web Application
 * Autonomous Agent Ecosystem - Modular Command Surface
 * 
 * Core application module providing:
 * - Client-side routing (SPA navigation)
 * - State management with reactive updates
 * - WebSocket connectivity for real-time data
 * - API client for backend communication
 * - UI components and interactions
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
  API_BASE_URL: window.location.origin,
  WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
  RECONNECT_INTERVAL: 3000,
  MAX_RECONNECT_ATTEMPTS: 10,
  LOG_BUFFER_SIZE: 500,
  TELEMETRY_UPDATE_INTERVAL: 2000,
  REQUEST_TIMEOUT: 30000, // 30 seconds
  API_VERSION: 'v1',
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

class StateManager {
  constructor() {
    this.state = {
      currentView: 'dashboard',
      systemStatus: 'offline',
      agents: [],
      workflows: [],
      tasks: [],
      messages: [],
      files: [],
      knowledgeItems: [],
      telemetry: {
        signalIntegrity: 0,
        bandwidth: 0,
        load: 0,
        latency: 0,
      },
      logs: [],
      settings: {
        insightBoost: true,
        safeMode: false,
        realtimeLogs: true,
        accentIntensity: 50,
      },
      user: null,
      wsConnected: false,
    };
    
    this.listeners = new Map();
    this.persistKeys = ['settings', 'user'];
    this.loadPersistedState();
  }

  loadPersistedState() {
    this.persistKeys.forEach(key => {
      const stored = localStorage.getItem(`astro_${key}`);
      if (stored) {
        try {
          this.state[key] = JSON.parse(stored);
        } catch (e) {
          console.warn(`Failed to parse persisted state for ${key}`, e);
        }
      }
    });
  }

  persistState(key) {
    if (this.persistKeys.includes(key)) {
      localStorage.setItem(`astro_${key}`, JSON.stringify(this.state[key]));
    }
  }

  get(key) {
    return key ? this.state[key] : this.state;
  }

  set(key, value) {
    const oldValue = this.state[key];
    this.state[key] = value;
    this.persistState(key);
    this.notify(key, value, oldValue);
  }

  update(key, updater) {
    const oldValue = this.state[key];
    const newValue = typeof updater === 'function' 
      ? updater(oldValue) 
      : { ...oldValue, ...updater };
    this.set(key, newValue);
  }

  subscribe(key, callback) {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set());
    }
    this.listeners.get(key).add(callback);
    return () => this.listeners.get(key).delete(callback);
  }

  notify(key, newValue, oldValue) {
    if (this.listeners.has(key)) {
      this.listeners.get(key).forEach(cb => cb(newValue, oldValue));
    }
    if (this.listeners.has('*')) {
      this.listeners.get('*').forEach(cb => cb(key, newValue, oldValue));
    }
  }
}

const state = new StateManager();

// ============================================================================
// ERROR CLASSES
// ============================================================================

class APIError extends Error {
  constructor(message, status, data = {}) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }

  isRateLimited() {
    return this.status === 429;
  }

  isUnauthorized() {
    return this.status === 401;
  }

  isServerError() {
    return this.status >= 500;
  }
}

// ============================================================================
// API CLIENT
// ============================================================================

class APIClient {
  constructor(baseUrl, timeout = CONFIG.REQUEST_TIMEOUT) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || this.timeout);
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: controller.signal,
      ...options,
    };

    // Remove custom properties before fetch
    delete config.timeout;

    if (config.body && typeof config.body === 'object') {
      config.body = JSON.stringify(config.body);
    }

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        const errorMessage = error.detail || `HTTP ${response.status}`;
        
        // Log rate limit info if present
        const rateLimitRemaining = response.headers.get('X-RateLimit-Remaining');
        if (rateLimitRemaining !== null) {
          console.warn(`Rate limit remaining: ${rateLimitRemaining}`);
        }
        
        throw new APIError(errorMessage, response.status, error);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        console.error(`API request timed out: ${endpoint}`);
        throw new APIError('Request timed out', 408, { detail: 'Request took too long' });
      }
      
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // System endpoints
  async getSystemStatus() {
    return this.request('/api/system/status');
  }

  async startSystem() {
    return this.request('/api/system/start', { method: 'POST' });
  }

  async stopSystem() {
    return this.request('/api/system/stop', { method: 'POST' });
  }

  // Agent endpoints
  async getAgents() {
    return this.request('/api/agents');
  }

  async getAgentStatus(agentId) {
    return this.request(`/api/agents/${agentId}/status`);
  }

  // Workflow endpoints
  async getWorkflows() {
    return this.request('/api/workflows');
  }

  async createWorkflow(workflow) {
    return this.request('/api/workflows', {
      method: 'POST',
      body: workflow,
    });
  }

  async getWorkflowStatus(workflowId) {
    return this.request(`/api/workflows/${workflowId}`);
  }

  // Chat/Command endpoints
  async sendCommand(command) {
    return this.request('/api/command', {
      method: 'POST',
      body: { command },
    });
  }

  async getConversationHistory(sessionId) {
    return this.request(`/api/chat/history/${sessionId}`);
  }

  async sendChatMessage(sessionId, message) {
    return this.request('/api/chat/message', {
      method: 'POST',
      body: { session_id: sessionId, message },
    });
  }

  // Knowledge vault endpoints
  async getKnowledgeItems(query = '') {
    const params = query ? `?q=${encodeURIComponent(query)}` : '';
    return this.request(`/api/knowledge${params}`);
  }

  async createKnowledgeItem(item) {
    return this.request('/api/knowledge', {
      method: 'POST',
      body: item,
    });
  }

  // File endpoints
  async getFiles(path = '/') {
    return this.request(`/api/files?path=${encodeURIComponent(path)}`);
  }

  async getFileContent(path) {
    return this.request(`/api/files/content?path=${encodeURIComponent(path)}`);
  }

  async createFile(path, content) {
    return this.request('/api/files', {
      method: 'POST',
      body: { path, content },
    });
  }

  // Telemetry
  async getTelemetry() {
    return this.request('/api/telemetry');
  }
}

const api = new APIClient(CONFIG.API_BASE_URL);

// ============================================================================
// WEBSOCKET CLIENT
// ============================================================================

class WebSocketClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.handlers = new Map();
    this.messageQueue = [];
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        state.set('wsConnected', true);
        this.reconnectAttempts = 0;
        this.flushMessageQueue();
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
        state.set('wsConnected', false);
        this.emit('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error', error);
        this.emit('error', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket', error);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= CONFIG.MAX_RECONNECT_ATTEMPTS) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = CONFIG.RECONNECT_INTERVAL * Math.pow(1.5, this.reconnectAttempts - 1);
    
    setTimeout(() => this.connect(), Math.min(delay, 30000));
  }

  handleMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case 'system_status':
        state.set('systemStatus', payload.status);
        break;

      case 'agent_update':
        state.update('agents', agents => 
          agents.map(a => a.id === payload.id ? { ...a, ...payload } : a)
        );
        break;

      case 'workflow_update':
        state.update('workflows', workflows =>
          workflows.map(w => w.id === payload.id ? { ...w, ...payload } : w)
        );
        break;

      case 'task_update':
        state.update('tasks', tasks => {
          const idx = tasks.findIndex(t => t.id === payload.id);
          if (idx >= 0) {
            tasks[idx] = { ...tasks[idx], ...payload };
          } else {
            tasks.push(payload);
          }
          return [...tasks];
        });
        break;

      case 'log':
        state.update('logs', logs => {
          const newLogs = [...logs, payload];
          return newLogs.slice(-CONFIG.LOG_BUFFER_SIZE);
        });
        break;

      case 'telemetry':
        state.set('telemetry', payload);
        break;

      case 'chat_message':
        state.update('messages', messages => [...messages, payload]);
        break;

      case 'file_change':
        this.emit('file_change', payload);
        break;

      default:
        this.emit(type, payload);
    }
  }

  send(type, payload) {
    const message = JSON.stringify({ type, payload });

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    } else {
      this.messageQueue.push(message);
    }
  }

  flushMessageQueue() {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(this.messageQueue.shift());
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

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

const ws = new WebSocketClient(CONFIG.WS_URL);

// ============================================================================
// ROUTER
// ============================================================================

class Router {
  constructor() {
    this.routes = new Map();
    this.currentRoute = null;
    this.viewContainer = null;
  }

  init(containerId) {
    this.viewContainer = document.getElementById(containerId);
    
    window.addEventListener('popstate', () => this.handleRoute());
    
    document.addEventListener('click', (e) => {
      const link = e.target.closest('[data-route]');
      if (link) {
        e.preventDefault();
        this.navigate(link.dataset.route);
      }
    });

    this.handleRoute();
  }

  register(path, handler) {
    this.routes.set(path, handler);
  }

  navigate(path, replace = false) {
    const method = replace ? 'replaceState' : 'pushState';
    history[method](null, '', path);
    this.handleRoute();
  }

  handleRoute() {
    const path = window.location.pathname || '/';
    const route = this.matchRoute(path);

    if (route) {
      this.currentRoute = route;
      state.set('currentView', route.name);
      route.handler(route.params);
    } else {
      this.navigate('/', true);
    }
  }

  matchRoute(path) {
    for (const [pattern, handler] of this.routes) {
      const params = this.matchPattern(pattern, path);
      if (params !== null) {
        return { pattern, handler, params, name: pattern.replace(/^\//, '') || 'dashboard' };
      }
    }
    return null;
  }

  matchPattern(pattern, path) {
    const patternParts = pattern.split('/').filter(Boolean);
    const pathParts = path.split('/').filter(Boolean);

    if (patternParts.length !== pathParts.length) {
      if (pattern === '/' && pathParts.length === 0) return {};
      return null;
    }

    const params = {};

    for (let i = 0; i < patternParts.length; i++) {
      if (patternParts[i].startsWith(':')) {
        params[patternParts[i].slice(1)] = pathParts[i];
      } else if (patternParts[i] !== pathParts[i]) {
        return null;
      }
    }

    return params;
  }
}

const router = new Router();

// ============================================================================
// VIEW RENDERERS
// ============================================================================

const Views = {
  dashboard() {
    return `
      <section class="dashboard-grid">
        ${Views.missionPanel()}
        ${Views.consolePanel()}
        ${Views.agentsPanel()}
        ${Views.telemetryPanel()}
        ${Views.workflowsPanel()}
        ${Views.controlsPanel()}
      </section>
      ${Views.systemPanel()}
    `;
  },

  missionPanel() {
    const workflows = state.get('workflows');
    const activeWorkflow = workflows.find(w => w.status === 'running') || workflows[0];
    
    if (!activeWorkflow) {
      return `
        <article class="panel mission">
          <div class="panel__header">
            <p class="eyebrow">Mission Directive</p>
          </div>
          <h2>No Active Mission</h2>
          <p>Start a new workflow or send a command to begin.</p>
        </article>
      `;
    }

    const progress = activeWorkflow.progress || 0;
    return `
      <article class="panel mission">
        <div class="panel__header">
          <p class="eyebrow">Mission Directive</p>
          <span class="badge badge--holo">${activeWorkflow.status}</span>
        </div>
        <h2>${activeWorkflow.name}</h2>
        <p>${activeWorkflow.description || ''}</p>
        <ul class="pill-list">
          ${(activeWorkflow.tags || []).map(tag => `<li>${tag}</li>`).join('')}
        </ul>
        <div class="progress">
          <span>${progress}% completion</span>
          <div class="progress__bar"><span style="width: ${progress}%"></span></div>
        </div>
      </article>
    `;
  },

  consolePanel() {
    const logs = state.get('logs').slice(-5);
    return `
      <article class="panel console">
        <div class="panel__header">
          <p class="eyebrow">Command Console</p>
          <span class="badge badge--dark" id="ws-status">${state.get('wsConnected') ? 'Live' : 'Offline'}</span>
        </div>
        <form id="command-form">
          <label class="console__label" for="command-input">Compose next instruction</label>
          <div class="console__input">
            <input
              id="command-input"
              name="command"
              type="text"
              placeholder="Research federated learning architectures..."
              autocomplete="off"
            />
            <button type="submit">Execute</button>
          </div>
        </form>
        <div class="console__log" id="console-log">
          ${logs.map(log => `<p><span>[${log.timestamp}]</span> ${log.message}</p>`).join('')}
        </div>
      </article>
    `;
  },

  agentsPanel() {
    const agents = state.get('agents');
    const statusMap = {
      online: 'status--online',
      busy: 'status--busy',
      idle: 'status--idle',
      offline: 'status--idle',
    };

    return `
      <article class="panel agents">
        <div class="panel__header">
          <p class="eyebrow">Agent Stack</p>
          <button class="micro-btn" data-action="refresh-agents">Refresh</button>
        </div>
        <ul class="agent-list">
          ${agents.length === 0 ? '<li><p>No agents registered</p></li>' : agents.map(agent => `
            <li data-agent-id="${agent.id}">
              <div>
                <span>${agent.icon || '🤖'} ${agent.name}</span>
                <small>${agent.description || agent.capabilities?.join(', ') || ''}</small>
              </div>
              <span class="status ${statusMap[agent.status] || 'status--idle'}">${agent.status}</span>
            </li>
          `).join('')}
        </ul>
      </article>
    `;
  },

  telemetryPanel() {
    const telemetry = state.get('telemetry');
    return `
      <article class="panel telemetry">
        <div class="panel__header">
          <p class="eyebrow">Telemetry Trace</p>
        </div>
        <div class="telemetry__chart">
          <canvas id="telemetry-canvas"></canvas>
          <div class="wave" aria-hidden="true"></div>
          <div class="wave wave--offset" aria-hidden="true"></div>
        </div>
        <ul class="telemetry__metrics">
          <li><span>Signal Integrity</span><strong>${telemetry.signalIntegrity?.toFixed(1) || 0}%</strong></li>
          <li><span>Bandwidth</span><strong>${telemetry.bandwidth?.toFixed(1) || 0} MB/s</strong></li>
          <li><span>Load</span><strong>${telemetry.load?.toFixed(0) || 0}%</strong></li>
        </ul>
      </article>
    `;
  },

  workflowsPanel() {
    const workflows = state.get('workflows').slice(0, 4);
    return `
      <article class="panel workflows">
        <div class="panel__header">
          <p class="eyebrow">Workflow Library</p>
          <button class="micro-btn" data-route="/workflows">Browse All</button>
        </div>
        <div class="workflow-grid">
          ${workflows.map(w => `
            <div data-workflow-id="${w.id}" data-action="view-workflow">
              <span>${w.name}</span>
              <small>${w.description || `${w.tasks?.length || 0} tasks`}</small>
            </div>
          `).join('')}
          ${workflows.length === 0 ? '<div><span>No workflows</span><small>Create one to get started</small></div>' : ''}
        </div>
      </article>
    `;
  },

  controlsPanel() {
    const settings = state.get('settings');
    return `
      <article class="panel clay">
        <p class="eyebrow">Tactile Controls</p>
        <div class="toggle-group">
          <label class="toggle">
            <input type="checkbox" data-setting="insightBoost" ${settings.insightBoost ? 'checked' : ''} />
            <span>Insight Boost</span>
          </label>
          <label class="toggle">
            <input type="checkbox" data-setting="safeMode" ${settings.safeMode ? 'checked' : ''} />
            <span>Safe Mode</span>
          </label>
          <label class="toggle">
            <input type="checkbox" data-setting="realtimeLogs" ${settings.realtimeLogs ? 'checked' : ''} />
            <span>Realtime Logs</span>
          </label>
        </div>
        <div class="knob" data-knob>
          <span>Accent Intensity</span>
          <div class="dial">
            <div class="dial__pointer" style="transform: rotate(${(settings.accentIntensity - 50) * 1.8}deg)"></div>
          </div>
        </div>
      </article>
    `;
  },

  systemPanel() {
    const logs = state.get('logs').slice(-3);
    return `
      <section class="system-panel">
        <div>
          <p class="eyebrow">Activity Ledger</p>
          <ol class="timeline" id="timeline">
            ${logs.map(log => `
              <li>
                <span>${log.timestamp}</span>
                <div>
                  <strong>${log.title || log.type}</strong>
                  <p>${log.message}</p>
                </div>
              </li>
            `).join('')}
            ${logs.length === 0 ? '<li><span>--:--</span><div><strong>Waiting</strong><p>System activity will appear here.</p></div></li>' : ''}
          </ol>
        </div>
        <div class="pillars">
          <article>
            <h3>System Status</h3>
            <p>Status: <strong>${state.get('systemStatus')}</strong></p>
            <p>WebSocket: <strong>${state.get('wsConnected') ? 'Connected' : 'Disconnected'}</strong></p>
          </article>
          <article>
            <h3>Active Agents</h3>
            <p>${state.get('agents').filter(a => a.status === 'online' || a.status === 'busy').length} / ${state.get('agents').length} online</p>
          </article>
          <article>
            <h3>Workflows</h3>
            <p>${state.get('workflows').filter(w => w.status === 'running').length} running</p>
          </article>
        </div>
      </section>
    `;
  },

  chat() {
    const messages = state.get('messages');
    return `
      <div class="chat-view">
        <header class="chat-header">
          <button class="back-btn" data-route="/">← Back</button>
          <h1>Chat Studio</h1>
          <span class="badge">${state.get('wsConnected') ? 'Live' : 'Offline'}</span>
        </header>
        <div class="chat-messages" id="chat-messages">
          ${messages.map(msg => `
            <div class="chat-message chat-message--${msg.role}">
              <div class="chat-message__avatar">${msg.role === 'user' ? '👤' : '🤖'}</div>
              <div class="chat-message__content">
                <p>${msg.content}</p>
                <span class="chat-message__time">${msg.timestamp || ''}</span>
              </div>
            </div>
          `).join('')}
          ${messages.length === 0 ? '<p class="chat-empty">Start a conversation with ASTRO agents...</p>' : ''}
        </div>
        <form class="chat-input" id="chat-form">
          <input type="text" id="chat-input" placeholder="Type your message..." autocomplete="off" />
          <button type="submit">Send</button>
        </form>
      </div>
    `;
  },

  workflows() {
    const workflows = state.get('workflows');
    return `
      <div class="workflows-view">
        <header class="view-header">
          <button class="back-btn" data-route="/">← Back</button>
          <h1>Workflow Canvas</h1>
          <button class="btn btn--primary" data-action="new-workflow">+ New Workflow</button>
        </header>
        <div class="workflows-list">
          ${workflows.map(w => `
            <article class="workflow-card" data-workflow-id="${w.id}">
              <div class="workflow-card__header">
                <h3>${w.name}</h3>
                <span class="status status--${w.status === 'running' ? 'busy' : w.status === 'completed' ? 'online' : 'idle'}">${w.status}</span>
              </div>
              <p>${w.description || ''}</p>
              <div class="workflow-card__meta">
                <span>${w.tasks?.length || 0} tasks</span>
                <span>${w.priority || 'medium'} priority</span>
              </div>
              <div class="workflow-card__actions">
                <button data-action="run-workflow" data-id="${w.id}">Run</button>
                <button data-action="view-workflow" data-id="${w.id}">View</button>
              </div>
            </article>
          `).join('')}
          ${workflows.length === 0 ? '<p class="empty-state">No workflows yet. Create one to get started.</p>' : ''}
        </div>
      </div>
    `;
  },

  vault() {
    const items = state.get('knowledgeItems');
    return `
      <div class="vault-view">
        <header class="view-header">
          <button class="back-btn" data-route="/">← Back</button>
          <h1>Knowledge Vault</h1>
          <div class="search-box">
            <input type="search" id="vault-search" placeholder="Search briefs..." />
          </div>
        </header>
        <div class="vault-grid">
          ${items.map(item => `
            <article class="vault-card" data-item-id="${item.id}">
              <div class="vault-card__type">${item.type || 'Document'}</div>
              <h3>${item.title}</h3>
              <p>${item.summary || item.content?.substring(0, 150) + '...' || ''}</p>
              <div class="vault-card__meta">
                <span>${item.created_at || ''}</span>
                ${item.tags?.map(t => `<span class="tag">${t}</span>`).join('') || ''}
              </div>
            </article>
          `).join('')}
          ${items.length === 0 ? '<p class="empty-state">Knowledge vault is empty. Research results will appear here.</p>' : ''}
        </div>
      </div>
    `;
  },

  files() {
    const files = state.get('files');
    return `
      <div class="files-view">
        <header class="view-header">
          <button class="back-btn" data-route="/">← Back</button>
          <h1>File Atelier</h1>
          <button class="btn btn--ghost" data-action="refresh-files">Refresh</button>
        </header>
        <div class="files-browser">
          <aside class="files-tree">
            <ul>
              ${files.filter(f => f.type === 'directory').map(dir => `
                <li class="folder" data-path="${dir.path}">
                  📁 ${dir.name}
                </li>
              `).join('')}
            </ul>
          </aside>
          <main class="files-list">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Size</th>
                  <th>Modified</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${files.filter(f => f.type === 'file').map(file => `
                  <tr data-path="${file.path}">
                    <td>📄 ${file.name}</td>
                    <td>${formatFileSize(file.size)}</td>
                    <td>${file.modified || ''}</td>
                    <td>
                      <button data-action="view-file" data-path="${file.path}">View</button>
                      <button data-action="download-file" data-path="${file.path}">Download</button>
                    </td>
                  </tr>
                `).join('')}
                ${files.filter(f => f.type === 'file').length === 0 ? '<tr><td colspan="4">No files in this directory</td></tr>' : ''}
              </tbody>
            </table>
          </main>
        </div>
      </div>
    `;
  },
};

// ============================================================================
// UTILITIES
// ============================================================================

function formatFileSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (bytes >= 1024 && i < units.length - 1) {
    bytes /= 1024;
    i++;
  }
  return `${bytes.toFixed(1)} ${units[i]}`;
}

function formatTimestamp(date = new Date()) {
  return date.toLocaleTimeString('en-US', { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: false 
  });
}

function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
}

function generateId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).substr(2, 9)}`;
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

const Handlers = {
  async handleCommandSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('command-input');
    const command = input.value.trim();
    
    if (!command) return;

    input.value = '';
    input.disabled = true;

    const log = {
      timestamp: formatTimestamp(),
      type: 'command',
      title: 'Command Sent',
      message: command,
    };
    state.update('logs', logs => [...logs, log]);

    try {
      const result = await api.sendCommand(command);
      state.update('logs', logs => [...logs, {
        timestamp: formatTimestamp(),
        type: 'response',
        title: 'Response',
        message: result.message || 'Command accepted',
      }]);
    } catch (error) {
      state.update('logs', logs => [...logs, {
        timestamp: formatTimestamp(),
        type: 'error',
        title: 'Error',
        message: error.message,
      }]);
    } finally {
      input.disabled = false;
      input.focus();
    }
  },

  async handleChatSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;

    input.value = '';

    const userMessage = {
      id: generateId(),
      role: 'user',
      content: message,
      timestamp: formatTimestamp(),
    };
    state.update('messages', msgs => [...msgs, userMessage]);

    try {
      const response = await api.sendChatMessage('default', message);
      const assistantMessage = {
        id: generateId(),
        role: 'assistant',
        content: response.message,
        timestamp: formatTimestamp(),
      };
      state.update('messages', msgs => [...msgs, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        id: generateId(),
        role: 'system',
        content: `Error: ${error.message}`,
        timestamp: formatTimestamp(),
      };
      state.update('messages', msgs => [...msgs, errorMessage]);
    }

    const container = document.getElementById('chat-messages');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  },

  handleSettingChange(e) {
    const setting = e.target.dataset.setting;
    if (setting) {
      state.update('settings', settings => ({
        ...settings,
        [setting]: e.target.checked,
      }));
    }
  },

  async handleAction(e) {
    const actionEl = e.target.closest('[data-action]');
    if (!actionEl) return;
    
    const action = actionEl.dataset.action;

    switch (action) {
      case 'start-system':
        await startSystem(actionEl);
        break;
      case 'stop-system':
        await stopSystem(actionEl);
        break;
      case 'refresh-agents':
        await loadAgents();
        break;
      case 'refresh-files':
        await loadFiles();
        break;
      case 'new-workflow':
        showWorkflowModal();
        break;
      case 'run-workflow':
        const workflowId = e.target.dataset.id;
        await api.request(`/api/workflows/${workflowId}/run`, { method: 'POST' });
        break;
      case 'view-file':
        const path = e.target.dataset.path;
        await showFileViewer(path);
        break;
    }
  },

  handleAccessCardClick(e) {
    const card = e.target.closest('.access-card');
    if (!card) return;

    const tag = card.querySelector('.tag')?.textContent;
    const routes = {
      'Chat Studio': '/chat',
      'Workflow Canvas': '/workflows',
      'Knowledge Vault': '/vault',
      'File Atelier': '/files',
    };

    if (routes[tag]) {
      router.navigate(routes[tag]);
    }
  },
};

// ============================================================================
// SYSTEM CONTROL
// ============================================================================

async function startSystem(button) {
  if (state.get('systemStatus') === 'online') {
    return;
  }

  const originalText = button.textContent;
  button.textContent = 'Deploying...';
  button.disabled = true;

  try {
    const result = await api.startSystem();
    state.set('systemStatus', 'online');
    
    button.textContent = 'System Online';
    button.dataset.action = 'stop-system';
    
    // Update hero stats
    updateHeroStats();
    
    state.update('logs', logs => [...logs, {
      timestamp: formatTimestamp(),
      type: 'system',
      title: 'System Started',
      message: 'ASTRO agent ecosystem is now online and ready.',
    }]);
  } catch (error) {
    button.textContent = originalText;
    state.update('logs', logs => [...logs, {
      timestamp: formatTimestamp(),
      type: 'error',
      title: 'Start Failed',
      message: error.message,
    }]);
  } finally {
    button.disabled = false;
  }
}

async function stopSystem(button) {
  const originalText = button.textContent;
  button.textContent = 'Stopping...';
  button.disabled = true;

  try {
    await api.stopSystem();
    state.set('systemStatus', 'offline');
    
    button.textContent = 'Deploy the Collective';
    button.dataset.action = 'start-system';
    
    updateHeroStats();
  } catch (error) {
    button.textContent = originalText;
  } finally {
    button.disabled = false;
  }
}

function updateHeroStats() {
  const agents = state.get('agents');
  const telemetry = state.get('telemetry');
  
  const agentsLive = document.querySelector('.hero__stats dd');
  if (agentsLive) {
    agentsLive.textContent = agents.filter(a => a.status !== 'offline').length.toString().padStart(2, '0');
  }
  
  const latencyEl = document.querySelectorAll('.hero__stats dd')[1];
  if (latencyEl) {
    latencyEl.textContent = `${telemetry.latency?.toFixed(1) || 1.4} s`;
  }
  
  const confidenceEl = document.querySelectorAll('.hero__stats dd')[2];
  if (confidenceEl) {
    const confidence = Math.round(telemetry.signalIntegrity || 97);
    confidenceEl.textContent = `${confidence}%`;
  }
}

// ============================================================================
// DATA LOADERS
// ============================================================================

async function loadAgents() {
  try {
    const agents = await api.getAgents();
    state.set('agents', agents);
  } catch (error) {
    console.error('Failed to load agents', error);
    state.set('agents', [
      { id: 'research_agent_001', name: 'Research · 001', icon: '🔬', status: 'idle', description: 'Deepcrawl / Contextual synthesis' },
      { id: 'code_agent_001', name: 'Code · 001', icon: '💻', status: 'idle', description: 'Python · QA harness' },
      { id: 'filesystem_agent_001', name: 'File · 001', icon: '📁', status: 'idle', description: 'Structured storage / Render' },
    ]);
  }
}

async function loadWorkflows() {
  try {
    const workflows = await api.getWorkflows();
    state.set('workflows', workflows);
  } catch (error) {
    console.error('Failed to load workflows', error);
    state.set('workflows', []);
  }
}

async function loadKnowledge() {
  try {
    const items = await api.getKnowledgeItems();
    state.set('knowledgeItems', items);
  } catch (error) {
    console.error('Failed to load knowledge items', error);
    state.set('knowledgeItems', []);
  }
}

async function loadFiles(path = '/') {
  try {
    const files = await api.getFiles(path);
    state.set('files', files);
  } catch (error) {
    console.error('Failed to load files', error);
    state.set('files', []);
  }
}

async function loadTelemetry() {
  try {
    const telemetry = await api.getTelemetry();
    state.set('telemetry', telemetry);
  } catch (error) {
    state.set('telemetry', {
      signalIntegrity: 99.3,
      bandwidth: 4.6,
      load: 31,
      latency: 1.4,
    });
  }
}

async function loadSystemStatus() {
  try {
    const status = await api.getSystemStatus();
    state.set('systemStatus', status.status);
  } catch (error) {
    state.set('systemStatus', 'offline');
  }
}

// ============================================================================
// UI HELPERS
// ============================================================================

function showWorkflowModal() {
  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal">
      <header class="modal__header">
        <h2>Create New Workflow</h2>
        <button class="modal__close" data-close>×</button>
      </header>
      <form id="workflow-form" class="modal__body">
        <div class="form-group">
          <label for="workflow-name">Name</label>
          <input type="text" id="workflow-name" required placeholder="My Research Workflow" />
        </div>
        <div class="form-group">
          <label for="workflow-desc">Description</label>
          <textarea id="workflow-desc" rows="3" placeholder="What should this workflow accomplish?"></textarea>
        </div>
        <div class="form-group">
          <label for="workflow-priority">Priority</label>
          <select id="workflow-priority">
            <option value="low">Low</option>
            <option value="medium" selected>Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <div class="modal__actions">
          <button type="button" class="btn btn--ghost" data-close>Cancel</button>
          <button type="submit" class="btn btn--primary">Create</button>
        </div>
      </form>
    </div>
  `;

  document.body.appendChild(modal);

  modal.addEventListener('click', (e) => {
    if (e.target === modal || e.target.closest('[data-close]')) {
      modal.remove();
    }
  });

  modal.querySelector('#workflow-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('workflow-name').value;
    const description = document.getElementById('workflow-desc').value;
    const priority = document.getElementById('workflow-priority').value;

    try {
      await api.createWorkflow({ name, description, priority, tasks: [] });
      await loadWorkflows();
      modal.remove();
    } catch (error) {
      alert('Failed to create workflow: ' + error.message);
    }
  });
}

async function showFileViewer(path) {
  try {
    const content = await api.getFileContent(path);
    
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
      <div class="modal modal--large">
        <header class="modal__header">
          <h2>${path.split('/').pop()}</h2>
          <button class="modal__close" data-close>×</button>
        </header>
        <div class="modal__body">
          <pre class="file-content"><code>${escapeHtml(content.content)}</code></pre>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    modal.addEventListener('click', (e) => {
      if (e.target === modal || e.target.closest('[data-close]')) {
        modal.remove();
      }
    });
  } catch (error) {
    alert('Failed to load file: ' + error.message);
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderView(html) {
  const container = document.getElementById('app-content') || document.querySelector('main');
  if (container) {
    container.innerHTML = html;
    bindEventListeners();
  }
}

function bindEventListeners() {
  const commandForm = document.getElementById('command-form');
  if (commandForm) {
    commandForm.addEventListener('submit', Handlers.handleCommandSubmit);
  }

  const chatForm = document.getElementById('chat-form');
  if (chatForm) {
    chatForm.addEventListener('submit', Handlers.handleChatSubmit);
  }

  document.querySelectorAll('[data-setting]').forEach(el => {
    el.addEventListener('change', Handlers.handleSettingChange);
  });

  document.querySelectorAll('[data-action]').forEach(el => {
    el.addEventListener('click', Handlers.handleAction);
  });

  document.querySelectorAll('.access-card').forEach(card => {
    card.addEventListener('click', Handlers.handleAccessCardClick);
  });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

async function initializeApp() {
  console.log('Initializing ASTRO Web Application...');

  router.register('/', () => renderView(Views.dashboard()));
  router.register('/chat', () => renderView(Views.chat()));
  router.register('/workflows', () => renderView(Views.workflows()));
  router.register('/vault', () => renderView(Views.vault()));
  router.register('/files', () => renderView(Views.files()));

  state.subscribe('*', debounce(() => {
    const view = state.get('currentView');
    const viewFn = Views[view] || Views.dashboard;
    renderView(viewFn());
  }, 100));

  await Promise.all([
    loadSystemStatus(),
    loadAgents(),
    loadWorkflows(),
    loadTelemetry(),
  ]);

  ws.connect();

  setInterval(loadTelemetry, CONFIG.TELEMETRY_UPDATE_INTERVAL);

  bindEventListeners();

  console.log('ASTRO Web Application initialized');
}

document.addEventListener('DOMContentLoaded', initializeApp);

// Export for debugging
window.ASTRO = { state, api, ws, router };
