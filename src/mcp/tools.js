/**
 * Built-in MCP Tools
 *
 * Browser-compatible tools for common operations
 */

import { createToolResult, ToolResultType } from './protocol.js'

/**
 * Web Fetch Tool - Fetch content from URLs
 */
export const WebFetchTool = {
  name: 'web_fetch',
  description: 'Fetch content from a URL. Supports text, JSON, and basic HTML parsing.',
  inputSchema: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        description: 'The URL to fetch'
      },
      method: {
        type: 'string',
        enum: ['GET', 'POST', 'PUT', 'DELETE'],
        default: 'GET'
      },
      headers: {
        type: 'object',
        description: 'Optional HTTP headers'
      },
      body: {
        type: 'string',
        description: 'Request body for POST/PUT'
      },
      parseAs: {
        type: 'string',
        enum: ['text', 'json', 'html'],
        default: 'text'
      }
    },
    required: ['url']
  },

  async handler({ url, method = 'GET', headers = {}, body, parseAs = 'text' }) {
    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? body : undefined
      })

      if (!response.ok) {
        return createToolResult(ToolResultType.ERROR, `HTTP ${response.status}: ${response.statusText}`)
      }

      let content

      switch (parseAs) {
        case 'json':
          content = await response.json()
          break
        case 'html':
          const html = await response.text()
          content = parseHTML(html)
          break
        default:
          content = await response.text()
      }

      return createToolResult(ToolResultType.TEXT, content, {
        url,
        status: response.status,
        contentType: response.headers.get('content-type')
      })
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, error.message)
    }
  }
}

/**
 * Simple HTML parser (extracts text and links)
 */
function parseHTML(html) {
  const parser = new DOMParser()
  const doc = parser.parseFromString(html, 'text/html')

  // Extract title
  const title = doc.querySelector('title')?.textContent || ''

  // Extract main content (simplified)
  const body = doc.body
  const textContent = body?.textContent?.replace(/\s+/g, ' ').trim() || ''

  // Extract links
  const links = Array.from(doc.querySelectorAll('a[href]'))
    .slice(0, 20)
    .map(a => ({ text: a.textContent?.trim(), href: a.href }))

  return { title, content: textContent.substring(0, 5000), links }
}

/**
 * Local Storage Tool - Persist data in browser
 */
export const StorageTool = {
  name: 'storage',
  description: 'Read and write data to browser local storage',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['get', 'set', 'delete', 'list', 'clear'],
        description: 'Storage operation'
      },
      key: {
        type: 'string',
        description: 'Storage key'
      },
      value: {
        type: 'string',
        description: 'Value to store (for set action)'
      },
      prefix: {
        type: 'string',
        description: 'Prefix for list action',
        default: 'mcp:'
      }
    },
    required: ['action']
  },

  async handler({ action, key, value, prefix = 'mcp:' }) {
    const fullKey = key ? `${prefix}${key}` : null

    switch (action) {
      case 'get':
        if (!fullKey) return createToolResult(ToolResultType.ERROR, 'Key required')
        const stored = localStorage.getItem(fullKey)
        return createToolResult(ToolResultType.TEXT, stored || null)

      case 'set':
        if (!fullKey) return createToolResult(ToolResultType.ERROR, 'Key required')
        localStorage.setItem(fullKey, value)
        return createToolResult(ToolResultType.TEXT, 'Stored successfully')

      case 'delete':
        if (!fullKey) return createToolResult(ToolResultType.ERROR, 'Key required')
        localStorage.removeItem(fullKey)
        return createToolResult(ToolResultType.TEXT, 'Deleted successfully')

      case 'list':
        const keys = []
        for (let i = 0; i < localStorage.length; i++) {
          const k = localStorage.key(i)
          if (k.startsWith(prefix)) {
            keys.push(k.substring(prefix.length))
          }
        }
        return createToolResult(ToolResultType.TEXT, keys)

      case 'clear':
        const toDelete = []
        for (let i = 0; i < localStorage.length; i++) {
          const k = localStorage.key(i)
          if (k.startsWith(prefix)) {
            toDelete.push(k)
          }
        }
        toDelete.forEach(k => localStorage.removeItem(k))
        return createToolResult(ToolResultType.TEXT, `Cleared ${toDelete.length} items`)

      default:
        return createToolResult(ToolResultType.ERROR, `Unknown action: ${action}`)
    }
  }
}

/**
 * IndexedDB Tool - Structured data storage
 */
export const DatabaseTool = {
  name: 'database',
  description: 'Store and query structured data using IndexedDB',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['get', 'put', 'delete', 'query', 'count'],
        description: 'Database operation'
      },
      store: {
        type: 'string',
        description: 'Object store name',
        default: 'default'
      },
      key: {
        type: 'string',
        description: 'Record key'
      },
      value: {
        type: 'object',
        description: 'Record value (for put)'
      },
      index: {
        type: 'string',
        description: 'Index name for queries'
      },
      range: {
        type: 'object',
        description: 'Query range { min, max }'
      }
    },
    required: ['action']
  },

  db: null,
  dbName: 'mcp-tools-db',

  async getDB() {
    if (this.db) return this.db

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1)

      request.onerror = () => reject(request.error)

      request.onupgradeneeded = (event) => {
        const db = event.target.result
        if (!db.objectStoreNames.contains('default')) {
          db.createObjectStore('default', { keyPath: 'id' })
        }
      }

      request.onsuccess = () => {
        this.db = request.result
        resolve(this.db)
      }
    })
  },

  async handler({ action, store = 'default', key, value, index, range }) {
    const db = await this.getDB()

    return new Promise((resolve) => {
      try {
        const tx = db.transaction([store], action === 'get' || action === 'query' || action === 'count' ? 'readonly' : 'readwrite')
        const objectStore = tx.objectStore(store)

        let request

        switch (action) {
          case 'get':
            request = objectStore.get(key)
            request.onsuccess = () => resolve(createToolResult(ToolResultType.TEXT, request.result))
            break

          case 'put':
            request = objectStore.put({ id: key, ...value, _updated: Date.now() })
            request.onsuccess = () => resolve(createToolResult(ToolResultType.TEXT, 'Stored'))
            break

          case 'delete':
            request = objectStore.delete(key)
            request.onsuccess = () => resolve(createToolResult(ToolResultType.TEXT, 'Deleted'))
            break

          case 'count':
            request = objectStore.count()
            request.onsuccess = () => resolve(createToolResult(ToolResultType.TEXT, request.result))
            break

          case 'query':
            const results = []
            request = objectStore.openCursor()
            request.onsuccess = (event) => {
              const cursor = event.target.result
              if (cursor) {
                results.push(cursor.value)
                cursor.continue()
              } else {
                resolve(createToolResult(ToolResultType.TEXT, results))
              }
            }
            break

          default:
            resolve(createToolResult(ToolResultType.ERROR, `Unknown action: ${action}`))
        }

        request.onerror = () => resolve(createToolResult(ToolResultType.ERROR, request.error?.message))
      } catch (error) {
        resolve(createToolResult(ToolResultType.ERROR, error.message))
      }
    })
  }
}

/**
 * Calculator Tool - Math operations
 */
export const CalculatorTool = {
  name: 'calculator',
  description: 'Perform mathematical calculations',
  inputSchema: {
    type: 'object',
    properties: {
      expression: {
        type: 'string',
        description: 'Math expression to evaluate (e.g., "2 + 2", "sqrt(16)", "sin(pi/2)")'
      }
    },
    required: ['expression']
  },

  async handler({ expression }) {
    try {
      // Safe math evaluation
      const math = {
        abs: Math.abs,
        sqrt: Math.sqrt,
        pow: Math.pow,
        log: Math.log,
        log10: Math.log10,
        exp: Math.exp,
        sin: Math.sin,
        cos: Math.cos,
        tan: Math.tan,
        asin: Math.asin,
        acos: Math.acos,
        atan: Math.atan,
        floor: Math.floor,
        ceil: Math.ceil,
        round: Math.round,
        min: Math.min,
        max: Math.max,
        pi: Math.PI,
        e: Math.E
      }

      // Replace function names and evaluate
      let expr = expression.toLowerCase()
      for (const [name, fn] of Object.entries(math)) {
        if (typeof fn === 'function') {
          expr = expr.replace(new RegExp(`\\b${name}\\b`, 'g'), `math.${name}`)
        } else {
          expr = expr.replace(new RegExp(`\\b${name}\\b`, 'g'), fn.toString())
        }
      }

      // Evaluate with restricted scope
      const result = new Function('math', `return ${expr}`)(math)

      return createToolResult(ToolResultType.TEXT, result)
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, `Invalid expression: ${error.message}`)
    }
  }
}

/**
 * DateTime Tool - Date and time operations
 */
export const DateTimeTool = {
  name: 'datetime',
  description: 'Get current date/time or parse/format dates',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['now', 'parse', 'format', 'diff', 'add'],
        default: 'now'
      },
      date: {
        type: 'string',
        description: 'Date string to parse'
      },
      format: {
        type: 'string',
        description: 'Output format',
        default: 'iso'
      },
      unit: {
        type: 'string',
        enum: ['days', 'hours', 'minutes', 'seconds'],
        description: 'Time unit for diff/add'
      },
      amount: {
        type: 'number',
        description: 'Amount to add (can be negative)'
      }
    }
  },

  async handler({ action = 'now', date, format = 'iso', unit, amount }) {
    try {
      const d = date ? new Date(date) : new Date()

      switch (action) {
        case 'now':
          return createToolResult(ToolResultType.TEXT, {
            iso: new Date().toISOString(),
            unix: Date.now(),
            local: new Date().toLocaleString()
          })

        case 'parse':
          return createToolResult(ToolResultType.TEXT, {
            iso: d.toISOString(),
            unix: d.getTime(),
            valid: !isNaN(d.getTime())
          })

        case 'format':
          let formatted
          switch (format) {
            case 'iso':
              formatted = d.toISOString()
              break
            case 'date':
              formatted = d.toDateString()
              break
            case 'time':
              formatted = d.toTimeString()
              break
            case 'local':
              formatted = d.toLocaleString()
              break
            default:
              formatted = d.toISOString()
          }
          return createToolResult(ToolResultType.TEXT, formatted)

        case 'diff':
          const now = new Date()
          const diffMs = now.getTime() - d.getTime()
          const multipliers = {
            seconds: 1000,
            minutes: 60000,
            hours: 3600000,
            days: 86400000
          }
          return createToolResult(ToolResultType.TEXT, diffMs / (multipliers[unit] || 1))

        case 'add':
          const multiplier = {
            seconds: 1000,
            minutes: 60000,
            hours: 3600000,
            days: 86400000
          }[unit] || 1
          const newDate = new Date(d.getTime() + (amount || 0) * multiplier)
          return createToolResult(ToolResultType.TEXT, newDate.toISOString())

        default:
          return createToolResult(ToolResultType.ERROR, `Unknown action: ${action}`)
      }
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, error.message)
    }
  }
}

/**
 * JSON Tool - Parse and manipulate JSON
 */
export const JSONTool = {
  name: 'json',
  description: 'Parse, query, and transform JSON data',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['parse', 'stringify', 'get', 'set', 'keys', 'values'],
        default: 'parse'
      },
      data: {
        type: 'string',
        description: 'JSON string or object'
      },
      path: {
        type: 'string',
        description: 'JSON path (e.g., "user.name" or "items[0].id")'
      },
      value: {
        description: 'Value to set'
      }
    },
    required: ['data']
  },

  async handler({ action = 'parse', data, path, value }) {
    try {
      let obj = typeof data === 'string' ? JSON.parse(data) : data

      switch (action) {
        case 'parse':
          return createToolResult(ToolResultType.TEXT, obj)

        case 'stringify':
          return createToolResult(ToolResultType.TEXT, JSON.stringify(obj, null, 2))

        case 'get':
          if (!path) return createToolResult(ToolResultType.TEXT, obj)
          const parts = path.replace(/\[(\d+)\]/g, '.$1').split('.')
          let result = obj
          for (const part of parts) {
            result = result?.[part]
          }
          return createToolResult(ToolResultType.TEXT, result)

        case 'set':
          const setParts = path.replace(/\[(\d+)\]/g, '.$1').split('.')
          let target = obj
          for (let i = 0; i < setParts.length - 1; i++) {
            target = target[setParts[i]]
          }
          target[setParts[setParts.length - 1]] = value
          return createToolResult(ToolResultType.TEXT, obj)

        case 'keys':
          return createToolResult(ToolResultType.TEXT, Object.keys(obj))

        case 'values':
          return createToolResult(ToolResultType.TEXT, Object.values(obj))

        default:
          return createToolResult(ToolResultType.ERROR, `Unknown action: ${action}`)
      }
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, error.message)
    }
  }
}

/**
 * Clipboard Tool - Read/write clipboard
 */
export const ClipboardTool = {
  name: 'clipboard',
  description: 'Read from or write to the system clipboard',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['read', 'write'],
        default: 'read'
      },
      text: {
        type: 'string',
        description: 'Text to write to clipboard'
      }
    }
  },

  async handler({ action = 'read', text }) {
    try {
      if (action === 'write') {
        await navigator.clipboard.writeText(text)
        return createToolResult(ToolResultType.TEXT, 'Copied to clipboard')
      } else {
        const content = await navigator.clipboard.readText()
        return createToolResult(ToolResultType.TEXT, content)
      }
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, `Clipboard error: ${error.message}`)
    }
  }
}

/**
 * Canvas/Image Tool - Basic image operations
 */
export const ImageTool = {
  name: 'image',
  description: 'Basic image operations (resize, convert, analyze)',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['analyze', 'resize', 'convert', 'crop'],
        default: 'analyze'
      },
      url: {
        type: 'string',
        description: 'Image URL or base64'
      },
      width: {
        type: 'number',
        description: 'Target width'
      },
      height: {
        type: 'number',
        description: 'Target height'
      },
      format: {
        type: 'string',
        enum: ['png', 'jpeg', 'webp'],
        default: 'png'
      }
    },
    required: ['url']
  },

  async handler({ action = 'analyze', url, width, height, format = 'png' }) {
    try {
      const img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = url
      })

      switch (action) {
        case 'analyze':
          return createToolResult(ToolResultType.TEXT, {
            width: img.naturalWidth,
            height: img.naturalHeight,
            aspectRatio: (img.naturalWidth / img.naturalHeight).toFixed(2)
          })

        case 'resize':
        case 'convert':
        case 'crop':
          const canvas = document.createElement('canvas')
          const ctx = canvas.getContext('2d')

          canvas.width = width || img.naturalWidth
          canvas.height = height || img.naturalHeight

          ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

          const mimeType = `image/${format}`
          const dataUrl = canvas.toDataURL(mimeType, 0.9)

          return createToolResult(ToolResultType.IMAGE, dataUrl, {
            width: canvas.width,
            height: canvas.height,
            format
          })

        default:
          return createToolResult(ToolResultType.ERROR, `Unknown action: ${action}`)
      }
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, error.message)
    }
  }
}

/**
 * Code Execution Tool - Run JavaScript safely
 */
export const CodeTool = {
  name: 'code',
  description: 'Execute JavaScript code in a sandboxed environment',
  inputSchema: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'JavaScript code to execute'
      },
      timeout: {
        type: 'number',
        description: 'Execution timeout in ms',
        default: 5000
      }
    },
    required: ['code']
  },

  async handler({ code, timeout = 5000 }) {
    try {
      // Create isolated context
      const iframe = document.createElement('iframe')
      iframe.style.display = 'none'
      iframe.sandbox = 'allow-scripts'
      document.body.appendChild(iframe)

      const result = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          reject(new Error('Execution timeout'))
        }, timeout)

        window.addEventListener('message', function handler(event) {
          if (event.source === iframe.contentWindow) {
            clearTimeout(timer)
            window.removeEventListener('message', handler)
            resolve(event.data)
          }
        })

        iframe.srcdoc = `
          <script>
            try {
              const result = eval(${JSON.stringify(code)});
              parent.postMessage({ success: true, result: JSON.stringify(result) }, '*');
            } catch (error) {
              parent.postMessage({ success: false, error: error.message }, '*');
            }
          </script>
        `
      })

      document.body.removeChild(iframe)

      if (result.success) {
        return createToolResult(ToolResultType.TEXT, JSON.parse(result.result || 'null'))
      } else {
        return createToolResult(ToolResultType.ERROR, result.error)
      }
    } catch (error) {
      return createToolResult(ToolResultType.ERROR, error.message)
    }
  }
}

/**
 * All built-in tools
 */
export const BuiltInTools = {
  web_fetch: WebFetchTool,
  storage: StorageTool,
  database: DatabaseTool,
  calculator: CalculatorTool,
  datetime: DateTimeTool,
  json: JSONTool,
  clipboard: ClipboardTool,
  image: ImageTool,
  code: CodeTool
}

/**
 * Register all built-in tools with an MCP client
 */
export function registerBuiltInTools(mcpClient) {
  for (const [name, tool] of Object.entries(BuiltInTools)) {
    mcpClient.registerLocalTool(tool)
  }
  return mcpClient
}

export default BuiltInTools
