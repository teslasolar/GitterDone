/**
 * Resource Manager
 *
 * Manage data resources that can be accessed by agents
 */

export class ResourceManager {
  constructor() {
    this.resources = new Map()
    this.subscriptions = new Map()
    this.cache = new Map()
    this.cacheTimeout = 60000 // 1 minute default
  }

  /**
   * Register a resource
   */
  register(uri, config) {
    this.resources.set(uri, {
      uri,
      name: config.name || uri,
      description: config.description || '',
      mimeType: config.mimeType || 'text/plain',
      loader: config.loader,
      schema: config.schema,
      cacheable: config.cacheable !== false,
      ttl: config.ttl || this.cacheTimeout
    })
    return this
  }

  /**
   * Read a resource
   */
  async read(uri, options = {}) {
    const resource = this.resources.get(uri)
    if (!resource) {
      throw new Error(`Resource not found: ${uri}`)
    }

    // Check cache
    if (resource.cacheable && !options.noCache) {
      const cached = this.cache.get(uri)
      if (cached && Date.now() - cached.timestamp < resource.ttl) {
        return cached.data
      }
    }

    // Load resource
    const data = await resource.loader(options)

    // Cache result
    if (resource.cacheable) {
      this.cache.set(uri, {
        data,
        timestamp: Date.now()
      })
    }

    // Notify subscribers
    this.notifySubscribers(uri, data)

    return data
  }

  /**
   * Subscribe to resource changes
   */
  subscribe(uri, callback) {
    if (!this.subscriptions.has(uri)) {
      this.subscriptions.set(uri, new Set())
    }
    this.subscriptions.get(uri).add(callback)

    return () => {
      this.subscriptions.get(uri)?.delete(callback)
    }
  }

  /**
   * Notify subscribers of changes
   */
  notifySubscribers(uri, data) {
    const subscribers = this.subscriptions.get(uri)
    if (subscribers) {
      for (const callback of subscribers) {
        callback(data)
      }
    }
  }

  /**
   * Invalidate cache
   */
  invalidate(uri) {
    if (uri) {
      this.cache.delete(uri)
    } else {
      this.cache.clear()
    }
  }

  /**
   * List all resources
   */
  list() {
    return Array.from(this.resources.values()).map(r => ({
      uri: r.uri,
      name: r.name,
      description: r.description,
      mimeType: r.mimeType
    }))
  }

  /**
   * Create a resource template
   */
  template(uriPattern, config) {
    return {
      pattern: uriPattern,
      config,
      resolve: (params) => {
        const uri = uriPattern.replace(/\{(\w+)\}/g, (_, key) => params[key] || '')
        return {
          ...config,
          uri,
          loader: () => config.loader(params)
        }
      }
    }
  }
}

/**
 * Built-in resource templates
 */
export const ResourceTemplates = {
  /**
   * URL resource - fetch from URL
   */
  url: {
    pattern: 'url://{path}',
    create: (url) => ({
      uri: `url://${encodeURIComponent(url)}`,
      name: url,
      mimeType: 'text/plain',
      loader: async () => {
        const response = await fetch(url)
        return response.text()
      }
    })
  },

  /**
   * LocalStorage resource
   */
  localStorage: {
    pattern: 'local://{key}',
    create: (key) => ({
      uri: `local://${key}`,
      name: key,
      mimeType: 'application/json',
      loader: async () => {
        const value = localStorage.getItem(key)
        return value ? JSON.parse(value) : null
      }
    })
  },

  /**
   * SessionStorage resource
   */
  sessionStorage: {
    pattern: 'session://{key}',
    create: (key) => ({
      uri: `session://${key}`,
      name: key,
      mimeType: 'application/json',
      loader: async () => {
        const value = sessionStorage.getItem(key)
        return value ? JSON.parse(value) : null
      }
    })
  },

  /**
   * IndexedDB resource
   */
  indexedDB: {
    pattern: 'idb://{database}/{store}/{key}',
    create: (database, store, key) => ({
      uri: `idb://${database}/${store}/${key}`,
      name: `${database}/${store}/${key}`,
      mimeType: 'application/json',
      loader: async () => {
        return new Promise((resolve, reject) => {
          const request = indexedDB.open(database)
          request.onerror = () => reject(request.error)
          request.onsuccess = () => {
            const db = request.result
            const tx = db.transaction(store, 'readonly')
            const objStore = tx.objectStore(store)
            const getRequest = objStore.get(key)
            getRequest.onsuccess = () => resolve(getRequest.result)
            getRequest.onerror = () => reject(getRequest.error)
          }
        })
      }
    })
  },

  /**
   * Clipboard resource
   */
  clipboard: {
    uri: 'clipboard://current',
    name: 'Clipboard',
    mimeType: 'text/plain',
    cacheable: false,
    loader: async () => {
      return navigator.clipboard.readText()
    }
  },

  /**
   * Time resource
   */
  time: {
    uri: 'time://now',
    name: 'Current Time',
    mimeType: 'application/json',
    cacheable: false,
    loader: async () => ({
      iso: new Date().toISOString(),
      unix: Date.now(),
      local: new Date().toLocaleString(),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    })
  },

  /**
   * Geolocation resource
   */
  geolocation: {
    uri: 'geo://current',
    name: 'Current Location',
    mimeType: 'application/json',
    cacheable: true,
    ttl: 300000, // 5 minutes
    loader: async () => {
      return new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          (pos) => resolve({
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
            accuracy: pos.coords.accuracy
          }),
          (err) => reject(new Error(err.message))
        )
      })
    }
  },

  /**
   * Browser info resource
   */
  browser: {
    uri: 'browser://info',
    name: 'Browser Info',
    mimeType: 'application/json',
    cacheable: true,
    ttl: Infinity,
    loader: async () => ({
      userAgent: navigator.userAgent,
      language: navigator.language,
      languages: navigator.languages,
      platform: navigator.platform,
      cookieEnabled: navigator.cookieEnabled,
      online: navigator.onLine,
      screen: {
        width: screen.width,
        height: screen.height,
        colorDepth: screen.colorDepth
      },
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      }
    })
  }
}

/**
 * Create a resource manager with built-in resources
 */
export function createResourceManager() {
  const manager = new ResourceManager()

  // Register built-in resources
  for (const [name, template] of Object.entries(ResourceTemplates)) {
    if (template.uri) {
      manager.register(template.uri, template)
    }
  }

  return manager
}

export default ResourceManager
