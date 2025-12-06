/**
 * ShardArray - Model weight sharding for Git LFS
 * Splits large models across multiple 50MB chunks for efficient loading
 */
export class ShardArray {
  constructor(baseUrl, shardSize = 50 * 1024 * 1024) { // 50MB default chunks
    this.baseUrl = baseUrl
    this.shardSize = shardSize
    this.cache = new Map() // shard index → Float32Array
    this.indexedDB = null
    this.dbName = 'model-weights'
    this.loading = new Map() // shard → Promise (prevent duplicate loads)
    this.metadata = null
  }

  /**
   * Initialize IndexedDB for persistent caching
   */
  async initCache() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => {
        this.indexedDB = request.result
        resolve()
      }

      request.onupgradeneeded = (event) => {
        const db = event.target.result
        if (!db.objectStoreNames.contains('shards')) {
          db.createObjectStore('shards', { keyPath: 'id' })
        }
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' })
        }
      }
    })
  }

  /**
   * Load model metadata (config.json)
   */
  async loadMetadata() {
    if (this.metadata) return this.metadata

    try {
      const response = await fetch(`${this.baseUrl}/config.json`)
      if (!response.ok) throw new Error(`Failed to load config: ${response.status}`)
      this.metadata = await response.json()
      return this.metadata
    } catch (error) {
      console.error('Failed to load model metadata:', error)
      throw error
    }
  }

  /**
   * Check if shard is in IndexedDB cache
   */
  async getCachedShard(shardIndex) {
    if (!this.indexedDB) return null

    return new Promise((resolve, reject) => {
      const tx = this.indexedDB.transaction('shards', 'readonly')
      const store = tx.objectStore('shards')
      const request = store.get(`${this.baseUrl}/shard_${shardIndex}`)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => {
        const result = request.result
        if (result) {
          resolve(new Float32Array(result.data))
        } else {
          resolve(null)
        }
      }
    })
  }

  /**
   * Store shard in IndexedDB cache
   */
  async cacheShard(shardIndex, data) {
    if (!this.indexedDB) return

    return new Promise((resolve, reject) => {
      const tx = this.indexedDB.transaction('shards', 'readwrite')
      const store = tx.objectStore('shards')
      const request = store.put({
        id: `${this.baseUrl}/shard_${shardIndex}`,
        data: data.buffer,
        timestamp: Date.now()
      })

      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve()
    })
  }

  /**
   * Load a single shard by index
   */
  async load(shardIndex, onProgress = null) {
    // Check memory cache
    if (this.cache.has(shardIndex)) {
      return this.cache.get(shardIndex)
    }

    // Check if already loading
    if (this.loading.has(shardIndex)) {
      return await this.loading.get(shardIndex)
    }

    // Start loading
    const loadPromise = this._loadShard(shardIndex, onProgress)
    this.loading.set(shardIndex, loadPromise)

    try {
      const data = await loadPromise
      this.cache.set(shardIndex, data)
      return data
    } finally {
      this.loading.delete(shardIndex)
    }
  }

  /**
   * Internal shard loading
   */
  async _loadShard(shardIndex, onProgress) {
    // Try IndexedDB cache first
    const cached = await this.getCachedShard(shardIndex)
    if (cached) {
      return cached
    }

    // Fetch from network
    const url = `${this.baseUrl}/shard_${String(shardIndex).padStart(3, '0')}.bin`
    const response = await fetch(url)

    if (!response.ok) {
      throw new Error(`Failed to load shard ${shardIndex}: ${response.status}`)
    }

    // Handle progress for large downloads
    if (onProgress && response.headers.get('content-length')) {
      const contentLength = parseInt(response.headers.get('content-length'))
      const reader = response.body.getReader()
      const chunks = []
      let receivedLength = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        chunks.push(value)
        receivedLength += value.length
        onProgress(receivedLength / contentLength)
      }

      const arrayBuffer = new Uint8Array(receivedLength)
      let position = 0
      for (const chunk of chunks) {
        arrayBuffer.set(chunk, position)
        position += chunk.length
      }

      const data = new Float32Array(arrayBuffer.buffer)
      await this.cacheShard(shardIndex, data)
      return data
    } else {
      const arrayBuffer = await response.arrayBuffer()
      const data = new Float32Array(arrayBuffer)
      await this.cacheShard(shardIndex, data)
      return data
    }
  }

  /**
   * Load multiple shards for a byte range
   */
  async loadRange(startOffset, endOffset, onProgress = null) {
    const startShard = Math.floor(startOffset / this.shardSize)
    const endShard = Math.floor(endOffset / this.shardSize)
    const shards = []

    const totalShards = endShard - startShard + 1
    let loadedShards = 0

    for (let i = startShard; i <= endShard; i++) {
      const data = await this.load(i, (progress) => {
        if (onProgress) {
          const overallProgress = (loadedShards + progress) / totalShards
          onProgress(overallProgress)
        }
      })
      shards.push({ index: i, data })
      loadedShards++
    }

    return this._extractRange(shards, startOffset, endOffset)
  }

  /**
   * Extract the exact byte range from loaded shards
   */
  _extractRange(shards, startOffset, endOffset) {
    const totalFloats = Math.ceil((endOffset - startOffset) / 4)
    const result = new Float32Array(totalFloats)
    let resultOffset = 0

    for (const { index, data } of shards) {
      const shardStart = index * this.shardSize / 4
      const shardEnd = shardStart + data.length

      const copyStart = Math.max(startOffset / 4, shardStart)
      const copyEnd = Math.min(endOffset / 4, shardEnd)

      if (copyStart < copyEnd) {
        const srcOffset = copyStart - shardStart
        const length = copyEnd - copyStart
        result.set(data.subarray(srcOffset, srcOffset + length), resultOffset)
        resultOffset += length
      }
    }

    return result
  }

  /**
   * Load weights for a specific layer
   */
  async loadLayer(layerName, onProgress = null) {
    const metadata = await this.loadMetadata()
    const layerInfo = metadata.layers?.[layerName]

    if (!layerInfo) {
      throw new Error(`Layer '${layerName}' not found in model metadata`)
    }

    return await this.loadRange(layerInfo.offset, layerInfo.offset + layerInfo.size, onProgress)
  }

  /**
   * Preload all shards
   */
  async preloadAll(onProgress = null) {
    const metadata = await this.loadMetadata()
    const totalShards = metadata.shards || 1

    for (let i = 0; i < totalShards; i++) {
      await this.load(i)
      if (onProgress) {
        onProgress((i + 1) / totalShards)
      }
    }
  }

  /**
   * Clear memory cache (keep IndexedDB)
   */
  clearMemoryCache() {
    this.cache.clear()
  }

  /**
   * Clear all caches including IndexedDB
   */
  async clearAllCaches() {
    this.cache.clear()

    if (this.indexedDB) {
      return new Promise((resolve, reject) => {
        const tx = this.indexedDB.transaction('shards', 'readwrite')
        const store = tx.objectStore('shards')
        const request = store.clear()

        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
      })
    }
  }

  /**
   * Get cache statistics
   */
  getStats() {
    let memorySize = 0
    for (const data of this.cache.values()) {
      memorySize += data.byteLength
    }

    return {
      memoryCachedShards: this.cache.size,
      memorySize: memorySize,
      shardSize: this.shardSize,
      baseUrl: this.baseUrl
    }
  }
}

export default ShardArray
