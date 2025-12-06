/**
 * Context Manager
 *
 * Manage conversation context, memory, and state for agents
 */

export class ContextManager {
  constructor(options = {}) {
    this.maxTokens = options.maxTokens || 4096
    this.maxMessages = options.maxMessages || 100
    this.maxToolResults = options.maxToolResults || 20

    this.messages = []
    this.toolResults = []
    this.variables = new Map()
    this.memory = new MemoryStore(options.memory || {})

    this.summarizer = options.summarizer || null
    this.tokenCounter = options.tokenCounter || defaultTokenCounter
  }

  /**
   * Add a message to context
   */
  addMessage(message) {
    this.messages.push({
      ...message,
      timestamp: Date.now(),
      id: generateId()
    })

    // Trim if needed
    this.trim()

    return this
  }

  /**
   * Add a tool result
   */
  addToolResult(toolName, result) {
    this.toolResults.push({
      tool: toolName,
      result,
      timestamp: Date.now(),
      id: generateId()
    })

    // Keep only recent tool results
    while (this.toolResults.length > this.maxToolResults) {
      this.toolResults.shift()
    }

    return this
  }

  /**
   * Set a context variable
   */
  set(key, value) {
    this.variables.set(key, value)
    return this
  }

  /**
   * Get a context variable
   */
  get(key) {
    return this.variables.get(key)
  }

  /**
   * Store in long-term memory
   */
  async remember(key, value, metadata = {}) {
    await this.memory.store(key, value, metadata)
    return this
  }

  /**
   * Recall from long-term memory
   */
  async recall(key) {
    return this.memory.retrieve(key)
  }

  /**
   * Search memory
   */
  async search(query, options = {}) {
    return this.memory.search(query, options)
  }

  /**
   * Get formatted context for prompts
   */
  format(options = {}) {
    const parts = []

    // System context
    if (this.variables.size > 0) {
      const vars = Object.fromEntries(this.variables)
      parts.push(`<context>\n${JSON.stringify(vars, null, 2)}\n</context>`)
    }

    // Recent tool results
    if (options.includeTools !== false && this.toolResults.length > 0) {
      const recentTools = this.toolResults.slice(-5)
      parts.push('<recent_tools>')
      for (const t of recentTools) {
        parts.push(`${t.tool}: ${JSON.stringify(t.result).substring(0, 500)}`)
      }
      parts.push('</recent_tools>')
    }

    // Messages
    for (const msg of this.messages) {
      switch (msg.role) {
        case 'system':
          parts.push(`<system>${msg.content}</system>`)
          break
        case 'user':
          parts.push(`<user>${msg.content}</user>`)
          break
        case 'assistant':
          parts.push(`<assistant>${msg.content}</assistant>`)
          break
        case 'tool':
          parts.push(`<tool name="${msg.name}">${msg.content}</tool>`)
          break
      }
    }

    return parts.join('\n\n')
  }

  /**
   * Get messages in OpenAI format
   */
  toMessages() {
    return this.messages.map(m => ({
      role: m.role,
      content: m.content,
      ...(m.name && { name: m.name })
    }))
  }

  /**
   * Trim context to fit within token limit
   */
  async trim() {
    // Remove old messages if over limit
    while (this.messages.length > this.maxMessages) {
      const removed = this.messages.shift()

      // Optionally summarize removed content
      if (this.summarizer && removed.role !== 'system') {
        await this.summarizeAndStore(removed)
      }
    }

    // Check token count
    let totalTokens = this.countTokens()

    while (totalTokens > this.maxTokens && this.messages.length > 2) {
      const removed = this.messages.splice(1, 1)[0] // Keep system message

      if (this.summarizer) {
        await this.summarizeAndStore(removed)
      }

      totalTokens = this.countTokens()
    }
  }

  /**
   * Count tokens in context
   */
  countTokens() {
    let total = 0
    for (const msg of this.messages) {
      total += this.tokenCounter(msg.content)
    }
    return total
  }

  /**
   * Summarize and store removed content
   */
  async summarizeAndStore(message) {
    if (!this.summarizer) return

    const summary = await this.summarizer(message.content)
    await this.memory.store(`msg_${message.id}`, {
      original: message,
      summary
    })
  }

  /**
   * Get a summary of the conversation
   */
  getSummary() {
    const userMessages = this.messages.filter(m => m.role === 'user').length
    const assistantMessages = this.messages.filter(m => m.role === 'assistant').length

    return {
      totalMessages: this.messages.length,
      userMessages,
      assistantMessages,
      toolCalls: this.toolResults.length,
      variables: this.variables.size,
      estimatedTokens: this.countTokens()
    }
  }

  /**
   * Clear context
   */
  clear() {
    this.messages = []
    this.toolResults = []
    this.variables.clear()
    return this
  }

  /**
   * Export context
   */
  export() {
    return {
      messages: this.messages,
      toolResults: this.toolResults,
      variables: Object.fromEntries(this.variables),
      timestamp: Date.now()
    }
  }

  /**
   * Import context
   */
  import(data) {
    this.messages = data.messages || []
    this.toolResults = data.toolResults || []
    this.variables = new Map(Object.entries(data.variables || {}))
    return this
  }

  /**
   * Fork context (create a copy)
   */
  fork() {
    const forked = new ContextManager({
      maxTokens: this.maxTokens,
      maxMessages: this.maxMessages
    })

    forked.messages = [...this.messages]
    forked.toolResults = [...this.toolResults]
    forked.variables = new Map(this.variables)

    return forked
  }
}

/**
 * Memory Store - Long-term storage for agent memory
 */
export class MemoryStore {
  constructor(options = {}) {
    this.backend = options.backend || 'indexeddb'
    this.namespace = options.namespace || 'agent-memory'
    this.db = null
    this.embedder = options.embedder || null // For semantic search
  }

  async init() {
    if (this.backend === 'indexeddb') {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open(this.namespace, 1)

        request.onerror = () => reject(request.error)

        request.onupgradeneeded = (event) => {
          const db = event.target.result
          if (!db.objectStoreNames.contains('memories')) {
            const store = db.createObjectStore('memories', { keyPath: 'key' })
            store.createIndex('timestamp', 'timestamp')
            store.createIndex('type', 'type')
          }
        }

        request.onsuccess = () => {
          this.db = request.result
          resolve()
        }
      })
    }
  }

  async store(key, value, metadata = {}) {
    if (!this.db) await this.init()

    const record = {
      key,
      value,
      metadata,
      timestamp: Date.now(),
      type: metadata.type || 'default'
    }

    // Generate embedding if embedder available
    if (this.embedder && typeof value === 'string') {
      record.embedding = await this.embedder(value)
    }

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['memories'], 'readwrite')
      const store = tx.objectStore('memories')
      const request = store.put(record)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  async retrieve(key) {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['memories'], 'readonly')
      const store = tx.objectStore('memories')
      const request = store.get(key)
      request.onsuccess = () => resolve(request.result?.value)
      request.onerror = () => reject(request.error)
    })
  }

  async search(query, options = {}) {
    if (!this.db) await this.init()

    const limit = options.limit || 10
    const type = options.type

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['memories'], 'readonly')
      const store = tx.objectStore('memories')

      const results = []
      let request

      if (type) {
        const index = store.index('type')
        request = index.openCursor(IDBKeyRange.only(type))
      } else {
        request = store.openCursor()
      }

      request.onsuccess = (event) => {
        const cursor = event.target.result
        if (cursor && results.length < limit) {
          const record = cursor.value

          // Simple text matching
          if (typeof record.value === 'string' &&
              record.value.toLowerCase().includes(query.toLowerCase())) {
            results.push(record)
          } else if (record.key.toLowerCase().includes(query.toLowerCase())) {
            results.push(record)
          }

          cursor.continue()
        } else {
          resolve(results)
        }
      }

      request.onerror = () => reject(request.error)
    })
  }

  async list(options = {}) {
    if (!this.db) await this.init()

    const limit = options.limit || 100

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['memories'], 'readonly')
      const store = tx.objectStore('memories')
      const index = store.index('timestamp')

      const results = []
      const request = index.openCursor(null, 'prev') // Most recent first

      request.onsuccess = (event) => {
        const cursor = event.target.result
        if (cursor && results.length < limit) {
          results.push(cursor.value)
          cursor.continue()
        } else {
          resolve(results)
        }
      }

      request.onerror = () => reject(request.error)
    })
  }

  async delete(key) {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['memories'], 'readwrite')
      const store = tx.objectStore('memories')
      const request = store.delete(key)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  async clear() {
    if (!this.db) await this.init()

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['memories'], 'readwrite')
      const store = tx.objectStore('memories')
      const request = store.clear()
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }
}

/**
 * Simple token counter (4 chars ≈ 1 token)
 */
function defaultTokenCounter(text) {
  if (!text) return 0
  return Math.ceil(text.length / 4)
}

/**
 * Generate unique ID
 */
function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}

/**
 * Create context manager with default settings
 */
export function createContext(options = {}) {
  return new ContextManager(options)
}

export default ContextManager
