/**
 * Model Context Protocol (MCP) Implementation
 *
 * Enables LLMs to interact with external tools, resources, and services
 * following the MCP specification for standardized tool calling.
 */

export class MCPClient {
  constructor(options = {}) {
    this.tools = new Map()
    this.resources = new Map()
    this.prompts = new Map()
    this.servers = new Map()
    this.pendingCalls = new Map()
    this.callId = 0

    this.options = {
      timeout: options.timeout || 30000,
      maxConcurrent: options.maxConcurrent || 10,
      retries: options.retries || 3,
      ...options
    }

    this.middleware = []
    this.eventHandlers = new Map()
  }

  /**
   * Connect to an MCP server
   */
  async connect(serverConfig) {
    const { name, transport, url, capabilities } = serverConfig

    let connection

    switch (transport) {
      case 'websocket':
        connection = await this.connectWebSocket(url)
        break
      case 'http':
        connection = await this.connectHTTP(url)
        break
      case 'postMessage':
        connection = await this.connectPostMessage(serverConfig.target)
        break
      case 'worker':
        connection = await this.connectWorker(url)
        break
      default:
        throw new Error(`Unknown transport: ${transport}`)
    }

    const server = {
      name,
      transport,
      connection,
      capabilities: capabilities || {},
      tools: new Map(),
      resources: new Map(),
      status: 'connected'
    }

    // Initialize and discover capabilities
    await this.initializeServer(server)

    this.servers.set(name, server)
    this.emit('server:connected', { name, server })

    return server
  }

  /**
   * WebSocket transport
   */
  async connectWebSocket(url) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        resolve({
          type: 'websocket',
          ws,
          send: (msg) => ws.send(JSON.stringify(msg)),
          close: () => ws.close()
        })
      }

      ws.onerror = reject

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        this.handleMessage(msg)
      }
    })
  }

  /**
   * HTTP transport
   */
  async connectHTTP(baseUrl) {
    return {
      type: 'http',
      baseUrl,
      send: async (msg) => {
        const response = await fetch(`${baseUrl}/mcp`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(msg)
        })
        return response.json()
      },
      close: () => {}
    }
  }

  /**
   * PostMessage transport (for iframes/windows)
   */
  async connectPostMessage(target) {
    const messageHandler = (event) => {
      if (event.data?.type === 'mcp') {
        this.handleMessage(event.data.payload)
      }
    }

    window.addEventListener('message', messageHandler)

    return {
      type: 'postMessage',
      target,
      send: (msg) => target.postMessage({ type: 'mcp', payload: msg }, '*'),
      close: () => window.removeEventListener('message', messageHandler)
    }
  }

  /**
   * Web Worker transport
   */
  async connectWorker(workerUrl) {
    const worker = new Worker(workerUrl, { type: 'module' })

    worker.onmessage = (event) => {
      this.handleMessage(event.data)
    }

    return {
      type: 'worker',
      worker,
      send: (msg) => worker.postMessage(msg),
      close: () => worker.terminate()
    }
  }

  /**
   * Initialize server and discover capabilities
   */
  async initializeServer(server) {
    const response = await this.sendRequest(server, 'initialize', {
      protocolVersion: '1.0',
      clientInfo: {
        name: 'WebGPU-Model-Runner',
        version: '1.0.0'
      },
      capabilities: {
        tools: true,
        resources: true,
        prompts: true,
        sampling: true
      }
    })

    server.serverInfo = response.serverInfo
    server.capabilities = response.capabilities

    // List available tools
    if (response.capabilities?.tools) {
      const toolsResponse = await this.sendRequest(server, 'tools/list', {})
      for (const tool of toolsResponse.tools || []) {
        server.tools.set(tool.name, tool)
        this.tools.set(`${server.name}:${tool.name}`, { ...tool, server: server.name })
      }
    }

    // List available resources
    if (response.capabilities?.resources) {
      const resourcesResponse = await this.sendRequest(server, 'resources/list', {})
      for (const resource of resourcesResponse.resources || []) {
        server.resources.set(resource.uri, resource)
        this.resources.set(resource.uri, { ...resource, server: server.name })
      }
    }

    // List available prompts
    if (response.capabilities?.prompts) {
      const promptsResponse = await this.sendRequest(server, 'prompts/list', {})
      for (const prompt of promptsResponse.prompts || []) {
        server.prompts.set(prompt.name, prompt)
        this.prompts.set(`${server.name}:${prompt.name}`, { ...prompt, server: server.name })
      }
    }
  }

  /**
   * Send request to server
   */
  async sendRequest(server, method, params) {
    const id = ++this.callId

    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingCalls.delete(id)
        reject(new Error(`Request timeout: ${method}`))
      }, this.options.timeout)

      this.pendingCalls.set(id, { resolve, reject, timeout })

      if (server.connection.type === 'http') {
        server.connection.send(request).then(response => {
          clearTimeout(timeout)
          this.pendingCalls.delete(id)
          if (response.error) {
            reject(new Error(response.error.message))
          } else {
            resolve(response.result)
          }
        }).catch(reject)
      } else {
        server.connection.send(request)
      }
    })
  }

  /**
   * Handle incoming message
   */
  handleMessage(msg) {
    if (msg.id && this.pendingCalls.has(msg.id)) {
      const { resolve, reject, timeout } = this.pendingCalls.get(msg.id)
      clearTimeout(timeout)
      this.pendingCalls.delete(msg.id)

      if (msg.error) {
        reject(new Error(msg.error.message))
      } else {
        resolve(msg.result)
      }
    } else if (msg.method) {
      // Handle server-initiated requests
      this.handleServerRequest(msg)
    }
  }

  /**
   * Handle server-initiated requests (sampling, etc.)
   */
  async handleServerRequest(msg) {
    const { method, params, id } = msg

    try {
      let result

      switch (method) {
        case 'sampling/createMessage':
          result = await this.handleSamplingRequest(params)
          break
        default:
          throw new Error(`Unknown method: ${method}`)
      }

      this.emit('server:request', { method, params, result })
    } catch (error) {
      this.emit('server:error', { method, error })
    }
  }

  /**
   * Call a tool
   */
  async callTool(toolName, args = {}) {
    // Run middleware
    for (const mw of this.middleware) {
      const result = await mw.beforeToolCall?.({ tool: toolName, args })
      if (result?.skip) return result.value
      if (result?.args) args = result.args
    }

    // Find tool
    let tool, serverName

    if (toolName.includes(':')) {
      [serverName, toolName] = toolName.split(':')
      tool = this.tools.get(`${serverName}:${toolName}`)
    } else {
      // Find first matching tool
      for (const [key, t] of this.tools) {
        if (key.endsWith(`:${toolName}`)) {
          tool = t
          serverName = t.server
          break
        }
      }
    }

    if (!tool) {
      throw new Error(`Tool not found: ${toolName}`)
    }

    const server = this.servers.get(serverName)
    if (!server) {
      throw new Error(`Server not connected: ${serverName}`)
    }

    this.emit('tool:calling', { tool: toolName, args })

    const result = await this.sendRequest(server, 'tools/call', {
      name: toolName,
      arguments: args
    })

    // Run after middleware
    for (const mw of this.middleware) {
      const modified = await mw.afterToolCall?.({ tool: toolName, args, result })
      if (modified?.result) return modified.result
    }

    this.emit('tool:result', { tool: toolName, result })

    return result
  }

  /**
   * Read a resource
   */
  async readResource(uri) {
    const resource = this.resources.get(uri)
    if (!resource) {
      throw new Error(`Resource not found: ${uri}`)
    }

    const server = this.servers.get(resource.server)

    const result = await this.sendRequest(server, 'resources/read', { uri })

    this.emit('resource:read', { uri, result })

    return result
  }

  /**
   * Get a prompt
   */
  async getPrompt(promptName, args = {}) {
    let prompt, serverName

    if (promptName.includes(':')) {
      [serverName, promptName] = promptName.split(':')
      prompt = this.prompts.get(`${serverName}:${promptName}`)
    } else {
      for (const [key, p] of this.prompts) {
        if (key.endsWith(`:${promptName}`)) {
          prompt = p
          serverName = p.server
          break
        }
      }
    }

    if (!prompt) {
      throw new Error(`Prompt not found: ${promptName}`)
    }

    const server = this.servers.get(serverName)

    const result = await this.sendRequest(server, 'prompts/get', {
      name: promptName,
      arguments: args
    })

    return result
  }

  /**
   * Register local tool (no server needed)
   */
  registerLocalTool(tool) {
    const { name, description, inputSchema, handler } = tool

    this.tools.set(`local:${name}`, {
      name,
      description,
      inputSchema,
      handler,
      server: 'local'
    })

    return this
  }

  /**
   * Add middleware
   */
  use(middleware) {
    this.middleware.push(middleware)
    return this
  }

  /**
   * Event handling
   */
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, [])
    }
    this.eventHandlers.get(event).push(handler)
    return this
  }

  emit(event, data) {
    const handlers = this.eventHandlers.get(event) || []
    for (const handler of handlers) {
      handler(data)
    }
  }

  /**
   * List all available tools
   */
  listTools() {
    return Array.from(this.tools.values()).map(t => ({
      name: t.name,
      description: t.description,
      server: t.server,
      inputSchema: t.inputSchema
    }))
  }

  /**
   * List all resources
   */
  listResources() {
    return Array.from(this.resources.values())
  }

  /**
   * Disconnect from server
   */
  disconnect(serverName) {
    const server = this.servers.get(serverName)
    if (server) {
      server.connection.close()

      // Remove tools and resources from this server
      for (const [key, tool] of this.tools) {
        if (tool.server === serverName) {
          this.tools.delete(key)
        }
      }

      this.servers.delete(serverName)
      this.emit('server:disconnected', { name: serverName })
    }
  }

  /**
   * Disconnect all servers
   */
  disconnectAll() {
    for (const name of this.servers.keys()) {
      this.disconnect(name)
    }
  }
}

/**
 * Tool call result types
 */
export const ToolResultType = {
  TEXT: 'text',
  IMAGE: 'image',
  RESOURCE: 'resource',
  ERROR: 'error'
}

/**
 * Create a tool result
 */
export function createToolResult(type, content, metadata = {}) {
  return {
    type,
    content,
    metadata,
    timestamp: Date.now()
  }
}

export default MCPClient
