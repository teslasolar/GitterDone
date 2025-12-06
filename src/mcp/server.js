/**
 * MCP Server Implementation
 *
 * Create custom MCP servers that can be connected to by agents
 */

export class MCPServer {
  constructor(options = {}) {
    this.name = options.name || 'custom-server'
    this.version = options.version || '1.0.0'

    this.tools = new Map()
    this.resources = new Map()
    this.prompts = new Map()

    this.connections = new Set()
    this.requestHandlers = new Map()

    this.capabilities = {
      tools: true,
      resources: true,
      prompts: true,
      sampling: options.sampling || false
    }

    // Register default handlers
    this.registerDefaultHandlers()
  }

  /**
   * Register default protocol handlers
   */
  registerDefaultHandlers() {
    this.requestHandlers.set('initialize', async (params) => {
      return {
        protocolVersion: '1.0',
        serverInfo: {
          name: this.name,
          version: this.version
        },
        capabilities: this.capabilities
      }
    })

    this.requestHandlers.set('tools/list', async () => {
      return {
        tools: Array.from(this.tools.values()).map(t => ({
          name: t.name,
          description: t.description,
          inputSchema: t.inputSchema
        }))
      }
    })

    this.requestHandlers.set('tools/call', async (params) => {
      const tool = this.tools.get(params.name)
      if (!tool) {
        throw new Error(`Unknown tool: ${params.name}`)
      }

      const result = await tool.handler(params.arguments || {})
      return { content: [result] }
    })

    this.requestHandlers.set('resources/list', async () => {
      return {
        resources: Array.from(this.resources.values()).map(r => ({
          uri: r.uri,
          name: r.name,
          description: r.description,
          mimeType: r.mimeType
        }))
      }
    })

    this.requestHandlers.set('resources/read', async (params) => {
      const resource = this.resources.get(params.uri)
      if (!resource) {
        throw new Error(`Unknown resource: ${params.uri}`)
      }

      const content = await resource.handler()
      return { contents: [content] }
    })

    this.requestHandlers.set('prompts/list', async () => {
      return {
        prompts: Array.from(this.prompts.values()).map(p => ({
          name: p.name,
          description: p.description,
          arguments: p.arguments
        }))
      }
    })

    this.requestHandlers.set('prompts/get', async (params) => {
      const prompt = this.prompts.get(params.name)
      if (!prompt) {
        throw new Error(`Unknown prompt: ${params.name}`)
      }

      const messages = await prompt.handler(params.arguments || {})
      return { messages }
    })
  }

  /**
   * Register a tool
   */
  tool(name, config) {
    this.tools.set(name, {
      name,
      description: config.description || '',
      inputSchema: config.inputSchema || { type: 'object', properties: {} },
      handler: config.handler
    })
    return this
  }

  /**
   * Register a resource
   */
  resource(uri, config) {
    this.resources.set(uri, {
      uri,
      name: config.name || uri,
      description: config.description || '',
      mimeType: config.mimeType || 'text/plain',
      handler: config.handler
    })
    return this
  }

  /**
   * Register a prompt template
   */
  prompt(name, config) {
    this.prompts.set(name, {
      name,
      description: config.description || '',
      arguments: config.arguments || [],
      handler: config.handler
    })
    return this
  }

  /**
   * Handle incoming request
   */
  async handleRequest(request) {
    const { method, params, id } = request

    const handler = this.requestHandlers.get(method)
    if (!handler) {
      return {
        jsonrpc: '2.0',
        id,
        error: { code: -32601, message: `Method not found: ${method}` }
      }
    }

    try {
      const result = await handler(params)
      return { jsonrpc: '2.0', id, result }
    } catch (error) {
      return {
        jsonrpc: '2.0',
        id,
        error: { code: -32000, message: error.message }
      }
    }
  }

  /**
   * Start WebSocket server (Node.js)
   */
  async listenWebSocket(port) {
    // This would use ws module in Node.js
    throw new Error('WebSocket server requires Node.js runtime')
  }

  /**
   * Create HTTP handler (for serverless/edge)
   */
  createHTTPHandler() {
    return async (req) => {
      if (req.method !== 'POST') {
        return new Response('Method not allowed', { status: 405 })
      }

      const body = await req.json()
      const response = await this.handleRequest(body)

      return new Response(JSON.stringify(response), {
        headers: { 'Content-Type': 'application/json' }
      })
    }
  }

  /**
   * Create Web Worker handler
   */
  createWorkerHandler() {
    return (event) => {
      this.handleRequest(event.data).then(response => {
        self.postMessage(response)
      })
    }
  }

  /**
   * Create PostMessage handler
   */
  createPostMessageHandler(targetOrigin = '*') {
    return (event) => {
      if (event.data?.type !== 'mcp') return

      this.handleRequest(event.data.payload).then(response => {
        event.source.postMessage({
          type: 'mcp',
          payload: response
        }, targetOrigin)
      })
    }
  }
}

/**
 * Create a server from a configuration object
 */
export function createServer(config) {
  const server = new MCPServer({
    name: config.name,
    version: config.version
  })

  // Register tools
  if (config.tools) {
    for (const [name, tool] of Object.entries(config.tools)) {
      server.tool(name, tool)
    }
  }

  // Register resources
  if (config.resources) {
    for (const [uri, resource] of Object.entries(config.resources)) {
      server.resource(uri, resource)
    }
  }

  // Register prompts
  if (config.prompts) {
    for (const [name, prompt] of Object.entries(config.prompts)) {
      server.prompt(name, prompt)
    }
  }

  return server
}

/**
 * Example server configuration
 */
export const ExampleServerConfig = {
  name: 'example-server',
  version: '1.0.0',

  tools: {
    echo: {
      description: 'Echo back the input',
      inputSchema: {
        type: 'object',
        properties: {
          message: { type: 'string', description: 'Message to echo' }
        },
        required: ['message']
      },
      handler: async ({ message }) => ({
        type: 'text',
        content: `Echo: ${message}`
      })
    },

    random: {
      description: 'Generate a random number',
      inputSchema: {
        type: 'object',
        properties: {
          min: { type: 'number', default: 0 },
          max: { type: 'number', default: 100 }
        }
      },
      handler: async ({ min = 0, max = 100 }) => ({
        type: 'text',
        content: String(Math.floor(Math.random() * (max - min + 1)) + min)
      })
    }
  },

  resources: {
    'config://app': {
      name: 'App Configuration',
      description: 'Application settings',
      mimeType: 'application/json',
      handler: async () => ({
        uri: 'config://app',
        mimeType: 'application/json',
        text: JSON.stringify({ theme: 'dark', language: 'en' })
      })
    }
  },

  prompts: {
    greet: {
      description: 'Generate a greeting',
      arguments: [
        { name: 'name', description: 'Name to greet', required: true }
      ],
      handler: async ({ name }) => [
        { role: 'user', content: `Please greet ${name} warmly.` }
      ]
    }
  }
}

export default MCPServer
