/**
 * API Server Template
 * OpenAI-compatible REST API for serving WebGPU models
 *
 * Run: node api-server.js --model ./models/my-model --port 3000
 */

const http = require('http')
const url = require('url')
const path = require('path')

// Configuration
const CONFIG = {
  port: parseInt(process.env.PORT || '3000'),
  modelPath: process.env.MODEL_PATH || './models/default',
  maxTokens: parseInt(process.env.MAX_TOKENS || '2048'),
  cors: process.env.CORS !== 'false',
  apiKey: process.env.API_KEY || null,
  rateLimit: parseInt(process.env.RATE_LIMIT || '60')
}

// Parse CLI args
process.argv.forEach((arg, i) => {
  if (arg === '--port' && process.argv[i + 1]) CONFIG.port = parseInt(process.argv[i + 1])
  if (arg === '--model' && process.argv[i + 1]) CONFIG.modelPath = process.argv[i + 1]
  if (arg === '--api-key' && process.argv[i + 1]) CONFIG.apiKey = process.argv[i + 1]
})

// Rate limiting
const rateLimiter = {
  requests: new Map(),

  check(ip) {
    const now = Date.now()
    const minuteAgo = now - 60000
    const reqs = this.requests.get(ip) || []
    const recent = reqs.filter(t => t > minuteAgo)

    if (recent.length >= CONFIG.rateLimit) {
      return false
    }

    recent.push(now)
    this.requests.set(ip, recent)
    return true
  }
}

// Mock model for template (replace with actual model loading)
const model = {
  loaded: false,
  config: null,

  async load(modelPath) {
    console.log(`Loading model from ${modelPath}...`)
    // In real implementation, load actual model
    this.config = {
      name: 'TemplateModel',
      version: '1.0.0',
      max_tokens: CONFIG.maxTokens
    }
    this.loaded = true
    console.log('Model loaded!')
  },

  async generate(prompt, options = {}) {
    // Mock generation - replace with actual inference
    const tokens = []
    const words = ['Hello', '!', ' ', 'This', ' ', 'is', ' ', 'a', ' ', 'response', '.']

    for (let i = 0; i < Math.min(options.max_tokens || 100, words.length); i++) {
      tokens.push(words[i])
      if (options.stream && options.onToken) {
        await options.onToken(words[i])
        await sleep(50) // Simulate generation delay
      }
    }

    return tokens.join('')
  },

  async embed(texts) {
    // Mock embeddings - replace with actual embedding
    return texts.map(text => {
      const embedding = new Array(384).fill(0).map(() => Math.random() - 0.5)
      return embedding
    })
  }
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms))
}

// Request handlers
const handlers = {
  // GET /v1/models
  async listModels(req, res) {
    res.json({
      object: 'list',
      data: [{
        id: model.config?.name || 'default',
        object: 'model',
        created: Date.now(),
        owned_by: 'local'
      }]
    })
  },

  // GET /v1/models/:model
  async getModel(req, res, modelId) {
    res.json({
      id: modelId,
      object: 'model',
      created: Date.now(),
      owned_by: 'local',
      ...model.config
    })
  },

  // POST /v1/completions
  async createCompletion(req, res, body) {
    const {
      prompt,
      max_tokens = 100,
      temperature = 0.7,
      top_p = 1,
      stream = false,
      stop = null
    } = body

    if (!prompt) {
      return res.error(400, 'prompt is required')
    }

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream')
      res.setHeader('Cache-Control', 'no-cache')
      res.setHeader('Connection', 'keep-alive')

      const output = await model.generate(prompt, {
        max_tokens,
        temperature,
        stream: true,
        onToken: (token) => {
          res.write(`data: ${JSON.stringify({
            id: `cmpl-${Date.now()}`,
            object: 'text_completion',
            choices: [{
              text: token,
              index: 0,
              finish_reason: null
            }]
          })}\n\n`)
        }
      })

      res.write(`data: ${JSON.stringify({
        id: `cmpl-${Date.now()}`,
        object: 'text_completion',
        choices: [{ text: '', index: 0, finish_reason: 'stop' }]
      })}\n\n`)
      res.write('data: [DONE]\n\n')
      res.end()
    } else {
      const output = await model.generate(prompt, { max_tokens, temperature })

      res.json({
        id: `cmpl-${Date.now()}`,
        object: 'text_completion',
        created: Math.floor(Date.now() / 1000),
        model: model.config?.name || 'default',
        choices: [{
          text: output,
          index: 0,
          logprobs: null,
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: prompt.length,
          completion_tokens: output.length,
          total_tokens: prompt.length + output.length
        }
      })
    }
  },

  // POST /v1/chat/completions
  async createChatCompletion(req, res, body) {
    const {
      messages,
      max_tokens = 100,
      temperature = 0.7,
      stream = false
    } = body

    if (!messages || !Array.isArray(messages)) {
      return res.error(400, 'messages is required and must be an array')
    }

    // Format messages into prompt
    const prompt = messages.map(m => {
      if (m.role === 'system') return `System: ${m.content}\n`
      if (m.role === 'user') return `User: ${m.content}\n`
      if (m.role === 'assistant') return `Assistant: ${m.content}\n`
      return ''
    }).join('') + 'Assistant: '

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream')
      res.setHeader('Cache-Control', 'no-cache')

      const output = await model.generate(prompt, {
        max_tokens,
        temperature,
        stream: true,
        onToken: (token) => {
          res.write(`data: ${JSON.stringify({
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            choices: [{
              index: 0,
              delta: { content: token },
              finish_reason: null
            }]
          })}\n\n`)
        }
      })

      res.write(`data: ${JSON.stringify({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
      })}\n\n`)
      res.write('data: [DONE]\n\n')
      res.end()
    } else {
      const output = await model.generate(prompt, { max_tokens, temperature })

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model.config?.name || 'default',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: output
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: prompt.length,
          completion_tokens: output.length,
          total_tokens: prompt.length + output.length
        }
      })
    }
  },

  // POST /v1/embeddings
  async createEmbeddings(req, res, body) {
    const { input, model: modelId } = body

    if (!input) {
      return res.error(400, 'input is required')
    }

    const texts = Array.isArray(input) ? input : [input]
    const embeddings = await model.embed(texts)

    res.json({
      object: 'list',
      data: embeddings.map((embedding, i) => ({
        object: 'embedding',
        embedding,
        index: i
      })),
      model: modelId || model.config?.name || 'default',
      usage: {
        prompt_tokens: texts.join('').length,
        total_tokens: texts.join('').length
      }
    })
  },

  // GET /health
  async health(req, res) {
    res.json({
      status: model.loaded ? 'healthy' : 'loading',
      model: model.config?.name || 'loading...',
      uptime: process.uptime()
    })
  }
}

// Create server
const server = http.createServer(async (req, res) => {
  // Add helpers
  res.json = (data) => {
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify(data))
  }

  res.error = (status, message) => {
    res.statusCode = status
    res.json({ error: { message, type: 'invalid_request_error' } })
  }

  // CORS
  if (CONFIG.cors) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization')
  }

  if (req.method === 'OPTIONS') {
    res.statusCode = 204
    return res.end()
  }

  // Rate limiting
  const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress
  if (!rateLimiter.check(clientIp)) {
    return res.error(429, 'Rate limit exceeded')
  }

  // API Key auth
  if (CONFIG.apiKey) {
    const authHeader = req.headers['authorization']
    if (!authHeader || authHeader !== `Bearer ${CONFIG.apiKey}`) {
      return res.error(401, 'Invalid API key')
    }
  }

  // Parse URL
  const parsed = url.parse(req.url, true)
  const pathname = parsed.pathname

  // Parse body for POST
  let body = {}
  if (req.method === 'POST') {
    body = await new Promise((resolve) => {
      let data = ''
      req.on('data', chunk => data += chunk)
      req.on('end', () => {
        try {
          resolve(JSON.parse(data))
        } catch {
          resolve({})
        }
      })
    })
  }

  // Route
  try {
    if (pathname === '/health' || pathname === '/') {
      return await handlers.health(req, res)
    }

    if (pathname === '/v1/models') {
      return await handlers.listModels(req, res)
    }

    if (pathname.startsWith('/v1/models/')) {
      const modelId = pathname.split('/')[3]
      return await handlers.getModel(req, res, modelId)
    }

    if (pathname === '/v1/completions' && req.method === 'POST') {
      return await handlers.createCompletion(req, res, body)
    }

    if (pathname === '/v1/chat/completions' && req.method === 'POST') {
      return await handlers.createChatCompletion(req, res, body)
    }

    if (pathname === '/v1/embeddings' && req.method === 'POST') {
      return await handlers.createEmbeddings(req, res, body)
    }

    res.error(404, 'Not found')
  } catch (err) {
    console.error('Request error:', err)
    res.error(500, err.message)
  }
})

// Start server
async function start() {
  await model.load(CONFIG.modelPath)

  server.listen(CONFIG.port, () => {
    console.log(`
╔════════════════════════════════════════════════════════╗
║           WebGPU Model API Server                      ║
╠════════════════════════════════════════════════════════╣
║  Endpoint: http://localhost:${String(CONFIG.port).padEnd(25)}║
║  Model: ${(model.config?.name || 'default').padEnd(33)}║
║  Auth: ${(CONFIG.apiKey ? 'Enabled' : 'Disabled').padEnd(34)}║
╠════════════════════════════════════════════════════════╣
║  Endpoints:                                            ║
║    GET  /health              - Health check            ║
║    GET  /v1/models           - List models             ║
║    POST /v1/completions      - Text completion         ║
║    POST /v1/chat/completions - Chat completion         ║
║    POST /v1/embeddings       - Generate embeddings     ║
╚════════════════════════════════════════════════════════╝

OpenAI-compatible. Use with any OpenAI SDK:

  import OpenAI from 'openai'
  const client = new OpenAI({
    baseURL: 'http://localhost:${CONFIG.port}/v1',
    apiKey: '${CONFIG.apiKey || 'not-required'}'
  })

`)
  })
}

start().catch(console.error)

module.exports = { server, handlers, model }
