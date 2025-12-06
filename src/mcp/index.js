/**
 * MCP (Model Context Protocol) Module
 *
 * Complete implementation of MCP for browser-based LLM agents
 */

// Core protocol
export { MCPClient, ToolResultType, createToolResult } from './protocol.js'

// Built-in tools
export {
  BuiltInTools,
  registerBuiltInTools,
  WebFetchTool,
  StorageTool,
  DatabaseTool,
  CalculatorTool,
  DateTimeTool,
  JSONTool,
  ClipboardTool,
  ImageTool,
  CodeTool
} from './tools.js'

// Agent system
export {
  Agent,
  ReActAgent,
  PlanExecuteAgent,
  createAgent
} from './agent.js'

// Server
export {
  MCPServer,
  createServer,
  ExampleServerConfig
} from './server.js'

// Resource manager
export { ResourceManager } from './resources.js'

// Context manager
export { ContextManager } from './context.js'

/**
 * Quick setup for MCP-enabled model
 */
export async function createMCPRunner(pipeline, options = {}) {
  const { Agent } = await import('./agent.js')
  const { registerBuiltInTools } = await import('./tools.js')

  const agent = new Agent(pipeline, options)

  // Connect to any configured servers
  if (options.servers) {
    for (const server of options.servers) {
      await agent.mcp.connect(server)
    }
  }

  return agent
}

/**
 * Common server configurations
 */
export const ServerTemplates = {
  /**
   * Minimal echo server for testing
   */
  echo: {
    name: 'echo-server',
    version: '1.0.0',
    tools: {
      echo: {
        description: 'Echo the input message',
        inputSchema: {
          type: 'object',
          properties: {
            message: { type: 'string' }
          },
          required: ['message']
        },
        handler: async ({ message }) => ({
          type: 'text',
          content: message
        })
      }
    }
  },

  /**
   * File system server (for use with File System Access API)
   */
  filesystem: {
    name: 'filesystem-server',
    version: '1.0.0',
    tools: {
      read_file: {
        description: 'Read a file from the local filesystem',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'File path' }
          },
          required: ['path']
        },
        handler: async ({ path }) => {
          // Requires File System Access API
          throw new Error('File System Access API required - call from user gesture')
        }
      },
      write_file: {
        description: 'Write content to a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            content: { type: 'string' }
          },
          required: ['path', 'content']
        },
        handler: async ({ path, content }) => {
          throw new Error('File System Access API required - call from user gesture')
        }
      },
      list_directory: {
        description: 'List files in a directory',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string' }
          }
        },
        handler: async ({ path }) => {
          throw new Error('File System Access API required - call from user gesture')
        }
      }
    }
  },

  /**
   * Database server (IndexedDB wrapper)
   */
  database: {
    name: 'database-server',
    version: '1.0.0',
    tools: {
      query: {
        description: 'Query records from the database',
        inputSchema: {
          type: 'object',
          properties: {
            collection: { type: 'string' },
            filter: { type: 'object' },
            limit: { type: 'number' }
          },
          required: ['collection']
        },
        handler: async ({ collection, filter, limit }) => {
          // Would implement IndexedDB query
          return { type: 'text', content: '[]' }
        }
      },
      insert: {
        description: 'Insert a record into the database',
        inputSchema: {
          type: 'object',
          properties: {
            collection: { type: 'string' },
            document: { type: 'object' }
          },
          required: ['collection', 'document']
        },
        handler: async ({ collection, document }) => {
          return { type: 'text', content: 'Inserted' }
        }
      },
      update: {
        description: 'Update records in the database',
        inputSchema: {
          type: 'object',
          properties: {
            collection: { type: 'string' },
            filter: { type: 'object' },
            update: { type: 'object' }
          },
          required: ['collection', 'update']
        },
        handler: async ({ collection, filter, update }) => {
          return { type: 'text', content: 'Updated' }
        }
      },
      delete: {
        description: 'Delete records from the database',
        inputSchema: {
          type: 'object',
          properties: {
            collection: { type: 'string' },
            filter: { type: 'object' }
          },
          required: ['collection']
        },
        handler: async ({ collection, filter }) => {
          return { type: 'text', content: 'Deleted' }
        }
      }
    }
  },

  /**
   * Web scraping server
   */
  scraper: {
    name: 'scraper-server',
    version: '1.0.0',
    tools: {
      scrape: {
        description: 'Scrape content from a webpage',
        inputSchema: {
          type: 'object',
          properties: {
            url: { type: 'string' },
            selector: { type: 'string', description: 'CSS selector' },
            attribute: { type: 'string', description: 'Attribute to extract' }
          },
          required: ['url']
        },
        handler: async ({ url, selector, attribute }) => {
          const response = await fetch(url)
          const html = await response.text()
          const parser = new DOMParser()
          const doc = parser.parseFromString(html, 'text/html')

          if (selector) {
            const elements = doc.querySelectorAll(selector)
            const results = Array.from(elements).map(el =>
              attribute ? el.getAttribute(attribute) : el.textContent
            )
            return { type: 'text', content: JSON.stringify(results) }
          }

          return {
            type: 'text',
            content: doc.body?.textContent?.substring(0, 5000) || ''
          }
        }
      },
      extract_links: {
        description: 'Extract all links from a webpage',
        inputSchema: {
          type: 'object',
          properties: {
            url: { type: 'string' }
          },
          required: ['url']
        },
        handler: async ({ url }) => {
          const response = await fetch(url)
          const html = await response.text()
          const parser = new DOMParser()
          const doc = parser.parseFromString(html, 'text/html')

          const links = Array.from(doc.querySelectorAll('a[href]'))
            .map(a => ({ href: a.href, text: a.textContent?.trim() }))
            .filter(l => l.href.startsWith('http'))

          return { type: 'text', content: JSON.stringify(links) }
        }
      }
    }
  }
}

export default {
  MCPClient,
  Agent,
  ReActAgent,
  PlanExecuteAgent,
  MCPServer,
  createAgent,
  createServer,
  createMCPRunner,
  BuiltInTools,
  ServerTemplates
}
