/**
 * ModelHub - Decentralized model registry and discovery
 * Share and discover WebGPU models via GitHub, IPFS, or custom registries
 */
export class ModelHub {
  constructor(options = {}) {
    this.registries = options.registries || [
      { name: 'github', type: 'github', baseUrl: 'https://api.github.com' },
      { name: 'local', type: 'local', baseUrl: null }
    ]

    this.cache = new Map()
    this.favorites = this.loadFavorites()
    this.downloadHistory = this.loadHistory()
  }

  /**
   * Search for models across all registries
   */
  async search(query, options = {}) {
    const {
      limit = 20,
      minStars = 0,
      maxSize = null,
      architecture = null,
      quantized = null
    } = options

    const results = []

    for (const registry of this.registries) {
      try {
        const models = await this.searchRegistry(registry, query, options)
        results.push(...models.map(m => ({ ...m, registry: registry.name })))
      } catch (error) {
        console.warn(`Failed to search ${registry.name}:`, error)
      }
    }

    // Sort by relevance/stars
    results.sort((a, b) => (b.stars || 0) - (a.stars || 0))

    return results.slice(0, limit)
  }

  /**
   * Search a specific registry
   */
  async searchRegistry(registry, query, options) {
    switch (registry.type) {
      case 'github':
        return this.searchGitHub(query, options)
      case 'ipfs':
        return this.searchIPFS(query, options)
      case 'local':
        return this.searchLocal(query, options)
      default:
        return this.searchCustom(registry, query, options)
    }
  }

  /**
   * Search GitHub for WebGPU models
   */
  async searchGitHub(query, options) {
    const searchQuery = `${query} webgpu-model in:readme,description`

    try {
      const response = await fetch(
        `https://api.github.com/search/repositories?q=${encodeURIComponent(searchQuery)}&sort=stars&per_page=20`
      )

      if (!response.ok) throw new Error('GitHub API error')

      const data = await response.json()

      // Filter and transform results
      const models = []
      for (const repo of data.items || []) {
        // Try to fetch model config
        const configUrl = `https://raw.githubusercontent.com/${repo.full_name}/main/config.json`
        try {
          const configResp = await fetch(configUrl)
          if (configResp.ok) {
            const config = await configResp.json()
            models.push({
              id: repo.full_name,
              name: config.name || repo.name,
              description: config.description || repo.description,
              url: `https://raw.githubusercontent.com/${repo.full_name}/main`,
              stars: repo.stargazers_count,
              size: config.parameters ? this.formatParams(config.parameters) : 'Unknown',
              sizeBytes: (config.shards || 1) * (config.shard_size || 50000000),
              architecture: config.architecture || 'transformer',
              quantized: config.dtype?.includes('int') || false,
              author: repo.owner.login,
              updated: repo.updated_at,
              config
            })
          }
        } catch (e) {
          // Skip repos without valid config
        }
      }

      return models
    } catch (error) {
      console.error('GitHub search failed:', error)
      return []
    }
  }

  /**
   * Search IPFS-based registry
   */
  async searchIPFS(query, options) {
    // IPFS-based model registry using DNSLink or IPNS
    const registryUrl = 'https://ipfs.io/ipns/webgpu-models.eth'

    try {
      const response = await fetch(`${registryUrl}/index.json`)
      if (!response.ok) return []

      const registry = await response.json()

      return registry.models
        .filter(m => m.name.toLowerCase().includes(query.toLowerCase()) ||
          m.description.toLowerCase().includes(query.toLowerCase()))
        .map(m => ({
          ...m,
          url: `https://ipfs.io/ipfs/${m.cid}`
        }))
    } catch (error) {
      return []
    }
  }

  /**
   * Search local/cached models
   */
  async searchLocal(query, options) {
    const models = []

    // Check IndexedDB for cached models
    try {
      const db = await this.openDB()
      const tx = db.transaction('models', 'readonly')
      const store = tx.objectStore('models')

      const allModels = await new Promise((resolve, reject) => {
        const request = store.getAll()
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error)
      })

      for (const model of allModels) {
        if (model.name.toLowerCase().includes(query.toLowerCase())) {
          models.push({
            ...model,
            registry: 'local',
            cached: true
          })
        }
      }
    } catch (error) {
      console.warn('Local search failed:', error)
    }

    return models
  }

  /**
   * Search custom registry
   */
  async searchCustom(registry, query, options) {
    try {
      const response = await fetch(`${registry.baseUrl}/search?q=${encodeURIComponent(query)}`)
      if (!response.ok) return []

      const data = await response.json()
      return data.models || []
    } catch (error) {
      return []
    }
  }

  /**
   * Get model details
   */
  async getModel(modelId, registry = 'github') {
    const cacheKey = `${registry}:${modelId}`
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)
    }

    let model
    switch (registry) {
      case 'github':
        model = await this.getGitHubModel(modelId)
        break
      case 'local':
        model = await this.getLocalModel(modelId)
        break
      default:
        model = await this.getCustomModel(modelId, registry)
    }

    if (model) {
      this.cache.set(cacheKey, model)
    }

    return model
  }

  /**
   * Get GitHub model details
   */
  async getGitHubModel(repoId) {
    const baseUrl = `https://raw.githubusercontent.com/${repoId}/main`

    try {
      // Fetch config
      const configResp = await fetch(`${baseUrl}/config.json`)
      if (!configResp.ok) throw new Error('Config not found')
      const config = await configResp.json()

      // Fetch tokenizer if available
      let tokenizer = null
      try {
        const tokResp = await fetch(`${baseUrl}/tokenizer.json`)
        if (tokResp.ok) tokenizer = await tokResp.json()
      } catch (e) { }

      // Fetch README if available
      let readme = null
      try {
        const readmeResp = await fetch(`${baseUrl}/README.md`)
        if (readmeResp.ok) readme = await readmeResp.text()
      } catch (e) { }

      return {
        id: repoId,
        url: baseUrl,
        config,
        tokenizer,
        readme,
        shards: config.shards || 1
      }
    } catch (error) {
      console.error('Failed to get GitHub model:', error)
      return null
    }
  }

  /**
   * Download and cache a model
   */
  async downloadModel(model, onProgress = null) {
    const db = await this.openDB()

    // Download config
    const configResp = await fetch(`${model.url}/config.json`)
    const config = await configResp.json()

    // Download tokenizer
    let tokenizer = null
    try {
      const tokResp = await fetch(`${model.url}/tokenizer.json`)
      if (tokResp.ok) tokenizer = await tokResp.json()
    } catch (e) { }

    // Download shards
    const shards = []
    const numShards = config.shards || 1

    for (let i = 0; i < numShards; i++) {
      const shardUrl = `${model.url}/shard_${String(i).padStart(3, '0')}.bin`
      const response = await fetch(shardUrl)

      if (!response.ok) {
        throw new Error(`Failed to download shard ${i}`)
      }

      const data = await response.arrayBuffer()
      shards.push(data)

      if (onProgress) {
        onProgress({
          phase: 'downloading',
          shard: i,
          total: numShards,
          progress: (i + 1) / numShards
        })
      }
    }

    // Store in IndexedDB
    const modelData = {
      id: model.id,
      name: config.name,
      url: model.url,
      config,
      tokenizer,
      shards,
      downloadedAt: Date.now()
    }

    const tx = db.transaction('models', 'readwrite')
    const store = tx.objectStore('models')
    await new Promise((resolve, reject) => {
      const request = store.put(modelData)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })

    // Update history
    this.addToHistory(model)

    return modelData
  }

  /**
   * Publish a model
   */
  async publishModel(config, weights, options = {}) {
    const {
      name,
      description,
      author,
      license = 'MIT',
      tags = []
    } = options

    // Validate
    if (!config.name || !weights) {
      throw new Error('Config and weights required')
    }

    // Generate model manifest
    const manifest = {
      name: config.name,
      version: config.version || '1.0.0',
      description,
      author,
      license,
      tags,
      architecture: config.architecture,
      parameters: config.parameters,
      vocab_size: config.vocab_size,
      embed_dim: config.embed_dim,
      shards: Math.ceil(weights.byteLength / (50 * 1024 * 1024)),
      created: new Date().toISOString()
    }

    // For GitHub publishing, we'd generate the repo structure
    // For now, return the publishable package
    return {
      manifest,
      files: {
        'config.json': JSON.stringify(config, null, 2),
        'manifest.json': JSON.stringify(manifest, null, 2),
        ...this.shardWeights(weights)
      }
    }
  }

  /**
   * Shard weights for publishing
   */
  shardWeights(weights) {
    const shardSize = 50 * 1024 * 1024
    const shards = {}
    let offset = 0
    let shardIdx = 0

    while (offset < weights.byteLength) {
      const chunk = weights.slice(offset, offset + shardSize)
      const name = `shard_${String(shardIdx).padStart(3, '0')}.bin`
      shards[name] = chunk
      offset += shardSize
      shardIdx++
    }

    return shards
  }

  /**
   * Rate/review a model
   */
  async rateModel(modelId, rating, review = null) {
    // Store locally for now
    const db = await this.openDB()
    const tx = db.transaction('ratings', 'readwrite')
    const store = tx.objectStore('ratings')

    await new Promise((resolve, reject) => {
      const request = store.put({
        modelId,
        rating,
        review,
        timestamp: Date.now()
      })
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })

    return true
  }

  /**
   * Add to favorites
   */
  toggleFavorite(modelId) {
    if (this.favorites.has(modelId)) {
      this.favorites.delete(modelId)
    } else {
      this.favorites.add(modelId)
    }
    this.saveFavorites()
    return this.favorites.has(modelId)
  }

  /**
   * Get popular models
   */
  async getPopular(limit = 10) {
    return this.search('', { limit, minStars: 10 })
  }

  /**
   * Get recently updated models
   */
  async getRecent(limit = 10) {
    const results = await this.search('')
    return results
      .sort((a, b) => new Date(b.updated) - new Date(a.updated))
      .slice(0, limit)
  }

  /**
   * Get recommended models based on history
   */
  async getRecommended(limit = 5) {
    // Simple recommendation based on past downloads
    const tags = new Set()
    for (const model of this.downloadHistory) {
      if (model.config?.tags) {
        model.config.tags.forEach(t => tags.add(t))
      }
    }

    if (tags.size === 0) {
      return this.getPopular(limit)
    }

    const query = [...tags].slice(0, 3).join(' ')
    return this.search(query, { limit })
  }

  // Helper methods
  formatParams(params) {
    if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`
    if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`
    if (params >= 1e3) return `${(params / 1e3).toFixed(1)}K`
    return params.toString()
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('model-hub', 1)
      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve(request.result)
      request.onupgradeneeded = (event) => {
        const db = event.target.result
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'id' })
        }
        if (!db.objectStoreNames.contains('ratings')) {
          db.createObjectStore('ratings', { keyPath: 'modelId' })
        }
      }
    })
  }

  loadFavorites() {
    try {
      const data = localStorage.getItem('model-favorites')
      return new Set(data ? JSON.parse(data) : [])
    } catch (e) {
      return new Set()
    }
  }

  saveFavorites() {
    localStorage.setItem('model-favorites', JSON.stringify([...this.favorites]))
  }

  loadHistory() {
    try {
      const data = localStorage.getItem('model-history')
      return data ? JSON.parse(data) : []
    } catch (e) {
      return []
    }
  }

  addToHistory(model) {
    this.downloadHistory.unshift({
      id: model.id,
      name: model.name,
      timestamp: Date.now()
    })
    this.downloadHistory = this.downloadHistory.slice(0, 50)
    localStorage.setItem('model-history', JSON.stringify(this.downloadHistory))
  }
}

export default ModelHub
