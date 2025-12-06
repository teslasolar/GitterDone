/**
 * Plugin System
 * Extensible architecture for custom model behaviors, processors, and integrations
 */

export class PluginManager {
  constructor() {
    this.plugins = new Map()
    this.hooks = new Map()
    this.processors = new Map()
    this.initialized = false
  }

  /**
   * Register a plugin
   */
  register(plugin) {
    if (!plugin.name) {
      throw new Error('Plugin must have a name')
    }

    if (this.plugins.has(plugin.name)) {
      console.warn(`Plugin ${plugin.name} already registered, replacing`)
    }

    this.plugins.set(plugin.name, plugin)

    // Register hooks
    if (plugin.hooks) {
      for (const [hookName, handler] of Object.entries(plugin.hooks)) {
        this.addHook(hookName, handler, plugin.name)
      }
    }

    // Register processors
    if (plugin.processors) {
      for (const [procName, processor] of Object.entries(plugin.processors)) {
        this.processors.set(`${plugin.name}:${procName}`, processor)
      }
    }

    console.log(`Plugin registered: ${plugin.name}`)
    return this
  }

  /**
   * Unregister a plugin
   */
  unregister(pluginName) {
    const plugin = this.plugins.get(pluginName)
    if (!plugin) return false

    // Remove hooks
    for (const [hookName, handlers] of this.hooks.entries()) {
      this.hooks.set(hookName, handlers.filter(h => h.plugin !== pluginName))
    }

    // Remove processors
    for (const key of this.processors.keys()) {
      if (key.startsWith(`${pluginName}:`)) {
        this.processors.delete(key)
      }
    }

    this.plugins.delete(pluginName)
    return true
  }

  /**
   * Add a hook handler
   */
  addHook(hookName, handler, pluginName = 'anonymous') {
    if (!this.hooks.has(hookName)) {
      this.hooks.set(hookName, [])
    }
    this.hooks.get(hookName).push({ handler, plugin: pluginName })
  }

  /**
   * Execute hook handlers
   */
  async executeHook(hookName, context = {}) {
    const handlers = this.hooks.get(hookName) || []
    let result = context

    for (const { handler, plugin } of handlers) {
      try {
        result = await handler(result) || result
      } catch (error) {
        console.error(`Hook error in ${plugin}:${hookName}:`, error)
      }
    }

    return result
  }

  /**
   * Get a processor
   */
  getProcessor(name) {
    return this.processors.get(name)
  }

  /**
   * Initialize all plugins
   */
  async init(context = {}) {
    for (const [name, plugin] of this.plugins) {
      if (plugin.init) {
        try {
          await plugin.init(context)
        } catch (error) {
          console.error(`Failed to initialize plugin ${name}:`, error)
        }
      }
    }
    this.initialized = true
  }

  /**
   * List all plugins
   */
  list() {
    return Array.from(this.plugins.entries()).map(([name, plugin]) => ({
      name,
      version: plugin.version || '1.0.0',
      description: plugin.description || '',
      hooks: plugin.hooks ? Object.keys(plugin.hooks) : [],
      processors: plugin.processors ? Object.keys(plugin.processors) : []
    }))
  }
}

// ============================================
// BUILT-IN PLUGINS
// ============================================

/**
 * Logging Plugin
 */
export const LoggingPlugin = {
  name: 'logging',
  version: '1.0.0',
  description: 'Log model inputs and outputs',

  config: {
    logInputs: true,
    logOutputs: true,
    logTokens: false,
    logPerformance: true
  },

  hooks: {
    'pre-generate': async (context) => {
      if (LoggingPlugin.config.logInputs) {
        console.log('[Input]', context.prompt?.substring(0, 100) + '...')
      }
      context._startTime = performance.now()
      return context
    },

    'post-generate': async (context) => {
      if (LoggingPlugin.config.logOutputs) {
        console.log('[Output]', context.output?.substring(0, 100) + '...')
      }
      if (LoggingPlugin.config.logPerformance) {
        const elapsed = performance.now() - context._startTime
        console.log(`[Performance] ${elapsed.toFixed(0)}ms`)
      }
      return context
    },

    'on-token': async (context) => {
      if (LoggingPlugin.config.logTokens) {
        process.stdout?.write(context.token) || console.log(context.token)
      }
      return context
    }
  }
}

/**
 * Safety Filter Plugin
 */
export const SafetyPlugin = {
  name: 'safety',
  version: '1.0.0',
  description: 'Filter unsafe content',

  config: {
    blocklist: [],
    maxOutputLength: 10000,
    blockPatterns: []
  },

  hooks: {
    'pre-generate': async (context) => {
      const { prompt } = context

      // Check blocklist
      for (const word of SafetyPlugin.config.blocklist) {
        if (prompt.toLowerCase().includes(word.toLowerCase())) {
          throw new Error('Input contains blocked content')
        }
      }

      // Check patterns
      for (const pattern of SafetyPlugin.config.blockPatterns) {
        if (new RegExp(pattern, 'i').test(prompt)) {
          throw new Error('Input matches blocked pattern')
        }
      }

      return context
    },

    'post-generate': async (context) => {
      let { output } = context

      // Truncate if too long
      if (output.length > SafetyPlugin.config.maxOutputLength) {
        output = output.substring(0, SafetyPlugin.config.maxOutputLength) + '...'
      }

      return { ...context, output }
    }
  }
}

/**
 * Caching Plugin
 */
export const CachingPlugin = {
  name: 'caching',
  version: '1.0.0',
  description: 'Cache model responses',

  cache: new Map(),
  maxSize: 1000,

  hooks: {
    'pre-generate': async (context) => {
      const key = CachingPlugin.hashPrompt(context.prompt)
      const cached = CachingPlugin.cache.get(key)

      if (cached) {
        console.log('[Cache] Hit')
        return { ...context, output: cached, cached: true }
      }

      context._cacheKey = key
      return context
    },

    'post-generate': async (context) => {
      if (context.cached) return context

      const key = context._cacheKey
      if (key && context.output) {
        // Evict oldest if full
        if (CachingPlugin.cache.size >= CachingPlugin.maxSize) {
          const firstKey = CachingPlugin.cache.keys().next().value
          CachingPlugin.cache.delete(firstKey)
        }

        CachingPlugin.cache.set(key, context.output)
      }

      return context
    }
  },

  hashPrompt(prompt) {
    let hash = 0
    for (let i = 0; i < prompt.length; i++) {
      const char = prompt.charCodeAt(i)
      hash = ((hash << 5) - hash) + char
      hash = hash & hash
    }
    return hash.toString(36)
  },

  clearCache() {
    this.cache.clear()
  }
}

/**
 * Rate Limiting Plugin
 */
export const RateLimitPlugin = {
  name: 'rate-limit',
  version: '1.0.0',
  description: 'Limit generation rate',

  config: {
    maxRequestsPerMinute: 60,
    maxTokensPerMinute: 10000
  },

  state: {
    requests: [],
    tokens: 0
  },

  hooks: {
    'pre-generate': async (context) => {
      const now = Date.now()
      const minuteAgo = now - 60000

      // Clean old requests
      RateLimitPlugin.state.requests = RateLimitPlugin.state.requests.filter(t => t > minuteAgo)

      if (RateLimitPlugin.state.requests.length >= RateLimitPlugin.config.maxRequestsPerMinute) {
        throw new Error('Rate limit exceeded')
      }

      RateLimitPlugin.state.requests.push(now)
      return context
    }
  }
}

/**
 * Metrics Plugin
 */
export const MetricsPlugin = {
  name: 'metrics',
  version: '1.0.0',
  description: 'Collect usage metrics',

  metrics: {
    totalRequests: 0,
    totalTokens: 0,
    totalTime: 0,
    errors: 0,
    requestsByHour: new Array(24).fill(0)
  },

  hooks: {
    'pre-generate': async (context) => {
      MetricsPlugin.metrics.totalRequests++
      const hour = new Date().getHours()
      MetricsPlugin.metrics.requestsByHour[hour]++
      return context
    },

    'on-token': async (context) => {
      MetricsPlugin.metrics.totalTokens++
      return context
    },

    'on-error': async (context) => {
      MetricsPlugin.metrics.errors++
      return context
    }
  },

  getMetrics() {
    return {
      ...this.metrics,
      avgTokensPerRequest: this.metrics.totalRequests > 0
        ? this.metrics.totalTokens / this.metrics.totalRequests
        : 0
    }
  },

  reset() {
    this.metrics = {
      totalRequests: 0,
      totalTokens: 0,
      totalTime: 0,
      errors: 0,
      requestsByHour: new Array(24).fill(0)
    }
  }
}

/**
 * Custom Prompt Template Plugin
 */
export const PromptTemplatePlugin = {
  name: 'prompt-template',
  version: '1.0.0',
  description: 'Apply prompt templates',

  templates: {
    default: '{prompt}',
    chat: '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n',
    llama: '[INST] {prompt} [/INST]',
    alpaca: '### Instruction:\n{prompt}\n\n### Response:\n',
    vicuna: 'USER: {prompt}\nASSISTANT: '
  },

  activeTemplate: 'default',

  hooks: {
    'pre-generate': async (context) => {
      const template = PromptTemplatePlugin.templates[PromptTemplatePlugin.activeTemplate]
      const formattedPrompt = template.replace('{prompt}', context.prompt)
      return { ...context, prompt: formattedPrompt }
    }
  },

  setTemplate(name) {
    if (this.templates[name]) {
      this.activeTemplate = name
    }
  },

  addTemplate(name, template) {
    this.templates[name] = template
  }
}

/**
 * Memory/Context Plugin
 */
export const ContextMemoryPlugin = {
  name: 'context-memory',
  version: '1.0.0',
  description: 'Maintain conversation context',

  config: {
    maxTurns: 10,
    maxTokens: 2000
  },

  history: [],

  hooks: {
    'pre-generate': async (context) => {
      // Build context from history
      let contextStr = ''
      for (const turn of ContextMemoryPlugin.history.slice(-ContextMemoryPlugin.config.maxTurns)) {
        contextStr += `User: ${turn.user}\nAssistant: ${turn.assistant}\n\n`
      }

      return {
        ...context,
        prompt: contextStr + 'User: ' + context.prompt + '\nAssistant: '
      }
    },

    'post-generate': async (context) => {
      ContextMemoryPlugin.history.push({
        user: context.originalPrompt || context.prompt,
        assistant: context.output,
        timestamp: Date.now()
      })

      // Trim history
      while (ContextMemoryPlugin.history.length > ContextMemoryPlugin.config.maxTurns) {
        ContextMemoryPlugin.history.shift()
      }

      return context
    }
  },

  clearHistory() {
    this.history = []
  },

  getHistory() {
    return [...this.history]
  }
}

/**
 * Create a custom plugin
 */
export function createPlugin(config) {
  return {
    name: config.name,
    version: config.version || '1.0.0',
    description: config.description || '',

    hooks: config.hooks || {},
    processors: config.processors || {},

    init: config.init,
    destroy: config.destroy
  }
}

// Global plugin manager instance
export const plugins = new PluginManager()

// Register built-in plugins
plugins.register(LoggingPlugin)
plugins.register(SafetyPlugin)
plugins.register(MetricsPlugin)

export default PluginManager
