/**
 * DistributedGPU - Distributed inference across browser tabs/workers
 * Uses BroadcastChannel and Web Workers for parallel model execution
 */
export class DistributedGPU {
  constructor(nodeId = null) {
    this.nodeId = nodeId || this.generateId()
    this.channel = null
    this.workers = []
    this.peers = new Map() // nodeId → last seen timestamp
    this.isCoordinator = false
    this.tasks = new Map() // taskId → { promise, resolve, reject }

    // Configuration
    this.heartbeatInterval = 5000
    this.peerTimeout = 15000

    // State
    this.gpu = null
    this.tensorCube = null
    this.ready = false
  }

  /**
   * Generate unique node ID
   */
  generateId() {
    return 'node_' + Math.random().toString(36).substr(2, 9)
  }

  /**
   * Initialize distributed system
   */
  async init(gpu, tensorCube) {
    this.gpu = gpu
    this.tensorCube = tensorCube

    // Setup broadcast channel for inter-tab communication
    this.channel = new BroadcastChannel('webgpu-distributed')
    this.channel.onmessage = (e) => this.handleMessage(e.data)

    // Announce presence
    this.broadcast({ type: 'join', nodeId: this.nodeId })

    // Start heartbeat
    this.heartbeatTimer = setInterval(() => this.heartbeat(), this.heartbeatInterval)

    // Cleanup old peers
    this.cleanupTimer = setInterval(() => this.cleanupPeers(), this.peerTimeout)

    // Elect coordinator after a short delay
    setTimeout(() => this.electCoordinator(), 1000)

    this.ready = true
    return this
  }

  /**
   * Handle incoming messages
   */
  handleMessage(msg) {
    switch (msg.type) {
      case 'join':
        this.handleJoin(msg)
        break
      case 'leave':
        this.handleLeave(msg)
        break
      case 'heartbeat':
        this.handleHeartbeat(msg)
        break
      case 'task':
        this.handleTask(msg)
        break
      case 'result':
        this.handleResult(msg)
        break
      case 'election':
        this.handleElection(msg)
        break
      case 'coordinator':
        this.handleCoordinator(msg)
        break
    }
  }

  /**
   * Handle peer join
   */
  handleJoin(msg) {
    if (msg.nodeId === this.nodeId) return
    this.peers.set(msg.nodeId, Date.now())

    // Respond with our presence
    this.broadcast({
      type: 'heartbeat',
      nodeId: this.nodeId,
      isCoordinator: this.isCoordinator
    })

    console.log(`[DistributedGPU] Peer joined: ${msg.nodeId}`)
  }

  /**
   * Handle peer leave
   */
  handleLeave(msg) {
    this.peers.delete(msg.nodeId)
    console.log(`[DistributedGPU] Peer left: ${msg.nodeId}`)

    // Re-elect if coordinator left
    if (msg.nodeId === this.coordinatorId) {
      this.electCoordinator()
    }
  }

  /**
   * Handle heartbeat
   */
  handleHeartbeat(msg) {
    if (msg.nodeId === this.nodeId) return
    this.peers.set(msg.nodeId, Date.now())

    if (msg.isCoordinator) {
      this.coordinatorId = msg.nodeId
    }
  }

  /**
   * Handle incoming task
   */
  async handleTask(msg) {
    if (msg.targetNode !== this.nodeId && msg.targetNode !== 'all') return

    try {
      const result = await this.executeTask(msg.task)
      this.broadcast({
        type: 'result',
        taskId: msg.taskId,
        sourceNode: this.nodeId,
        data: result
      })
    } catch (error) {
      this.broadcast({
        type: 'result',
        taskId: msg.taskId,
        sourceNode: this.nodeId,
        error: error.message
      })
    }
  }

  /**
   * Handle task result
   */
  handleResult(msg) {
    const task = this.tasks.get(msg.taskId)
    if (!task) return

    if (msg.error) {
      task.errors = task.errors || []
      task.errors.push({ node: msg.sourceNode, error: msg.error })
    } else {
      task.results = task.results || []
      task.results.push({ node: msg.sourceNode, data: msg.data })
    }

    // Check if all results received
    if (task.results?.length >= task.expectedResults) {
      task.resolve(task.results)
      this.tasks.delete(msg.taskId)
    }
  }

  /**
   * Execute a compute task locally
   */
  async executeTask(task) {
    switch (task.type) {
      case 'matmul':
        return await this.tensorCube.matmul(
          task.inputA, task.inputB, task.output,
          task.M, task.N, task.K
        )

      case 'forward':
        // Forward pass on a subset of data
        const input = new Float32Array(task.inputData)
        this.tensorCube.createTensor('task_input', input, task.inputShape)
        // ... execute layers
        return await this.tensorCube.readTensor('task_output')

      default:
        throw new Error(`Unknown task type: ${task.type}`)
    }
  }

  /**
   * Handle coordinator election
   */
  handleElection(msg) {
    // Bully algorithm - highest ID wins
    if (msg.nodeId > this.nodeId) {
      // Someone with higher ID is alive, they'll be coordinator
      this.isCoordinator = false
    } else if (msg.nodeId < this.nodeId) {
      // We have higher ID, announce victory
      this.broadcast({ type: 'election', nodeId: this.nodeId })
    }
  }

  /**
   * Handle coordinator announcement
   */
  handleCoordinator(msg) {
    this.coordinatorId = msg.nodeId
    this.isCoordinator = (msg.nodeId === this.nodeId)
    console.log(`[DistributedGPU] Coordinator: ${msg.nodeId}${this.isCoordinator ? ' (me)' : ''}`)
  }

  /**
   * Elect coordinator
   */
  electCoordinator() {
    // Get all known nodes including self
    const allNodes = [this.nodeId, ...this.peers.keys()].sort()

    // Highest ID becomes coordinator
    const coordinator = allNodes[allNodes.length - 1]
    this.coordinatorId = coordinator
    this.isCoordinator = (coordinator === this.nodeId)

    if (this.isCoordinator) {
      this.broadcast({ type: 'coordinator', nodeId: this.nodeId })
      console.log('[DistributedGPU] I am the coordinator')
    }
  }

  /**
   * Send heartbeat
   */
  heartbeat() {
    this.broadcast({
      type: 'heartbeat',
      nodeId: this.nodeId,
      isCoordinator: this.isCoordinator
    })
  }

  /**
   * Cleanup stale peers
   */
  cleanupPeers() {
    const now = Date.now()
    for (const [nodeId, lastSeen] of this.peers.entries()) {
      if (now - lastSeen > this.peerTimeout) {
        this.peers.delete(nodeId)
        console.log(`[DistributedGPU] Peer timed out: ${nodeId}`)

        if (nodeId === this.coordinatorId) {
          this.electCoordinator()
        }
      }
    }
  }

  /**
   * Broadcast message to all peers
   */
  broadcast(msg) {
    if (this.channel) {
      this.channel.postMessage(msg)
    }
  }

  /**
   * Distribute a tensor across peers for parallel processing
   */
  async scatter(tensor, dim = 0) {
    const numPeers = this.peers.size + 1 // Include self
    const chunks = this.shard(tensor, numPeers, dim)

    // Send chunks to peers
    const peerIds = [this.nodeId, ...this.peers.keys()]
    const taskId = this.generateId()

    const promises = peerIds.map((peerId, i) => {
      return new Promise((resolve, reject) => {
        if (peerId === this.nodeId) {
          // Local chunk
          resolve({ node: this.nodeId, data: chunks[i] })
        } else {
          // Remote chunk
          this.broadcast({
            type: 'task',
            taskId: taskId + '_' + i,
            targetNode: peerId,
            task: {
              type: 'store',
              data: Array.from(chunks[i])
            }
          })

          // Store promise for later resolution
          this.tasks.set(taskId + '_' + i, { resolve, reject, expectedResults: 1 })
        }
      })
    })

    return await Promise.all(promises)
  }

  /**
   * Gather results from all peers
   */
  async gather(taskId) {
    return new Promise((resolve, reject) => {
      const numPeers = this.peers.size + 1

      this.tasks.set(taskId, {
        resolve,
        reject,
        expectedResults: numPeers,
        results: []
      })

      // Request results from all peers
      this.broadcast({
        type: 'task',
        taskId,
        targetNode: 'all',
        task: { type: 'get_result' }
      })

      // Timeout
      setTimeout(() => {
        const task = this.tasks.get(taskId)
        if (task && task.results.length < numPeers) {
          resolve(task.results) // Return partial results
          this.tasks.delete(taskId)
        }
      }, 10000)
    })
  }

  /**
   * Shard tensor along dimension
   */
  shard(tensor, numShards, dim = 0) {
    const shape = tensor.shape || [tensor.length]
    const size = shape[dim] || tensor.length
    const chunkSize = Math.ceil(size / numShards)

    const chunks = []
    for (let i = 0; i < numShards; i++) {
      const start = i * chunkSize
      const end = Math.min(start + chunkSize, size)

      if (dim === 0 && shape.length === 1) {
        chunks.push(tensor.slice(start, end))
      } else {
        // Multi-dimensional slicing
        chunks.push(this.sliceTensor(tensor, dim, start, end))
      }
    }

    return chunks
  }

  /**
   * Slice tensor along dimension
   */
  sliceTensor(tensor, dim, start, end) {
    // Simplified for 1D/2D tensors
    if (!tensor.shape || tensor.shape.length <= 1) {
      return tensor.slice(start, end)
    }

    const [rows, cols] = tensor.shape
    if (dim === 0) {
      return tensor.slice(start * cols, end * cols)
    } else {
      // Column slicing - more complex
      const result = new Float32Array((end - start) * rows)
      for (let r = 0; r < rows; r++) {
        for (let c = start; c < end; c++) {
          result[r * (end - start) + (c - start)] = tensor[r * cols + c]
        }
      }
      return result
    }
  }

  /**
   * Concatenate results from gather
   */
  concat(results, dim = 0) {
    if (results.length === 0) return new Float32Array(0)
    if (results.length === 1) return results[0].data

    // Simple concatenation for 1D
    const totalSize = results.reduce((sum, r) => sum + r.data.length, 0)
    const output = new Float32Array(totalSize)

    let offset = 0
    for (const result of results) {
      output.set(result.data, offset)
      offset += result.data.length
    }

    return output
  }

  /**
   * Get cluster info
   */
  getClusterInfo() {
    return {
      nodeId: this.nodeId,
      isCoordinator: this.isCoordinator,
      coordinatorId: this.coordinatorId,
      numPeers: this.peers.size,
      peers: Array.from(this.peers.keys()),
      ready: this.ready
    }
  }

  /**
   * Spawn a Web Worker for local parallelism
   */
  async spawnWorker() {
    const workerCode = `
      self.onmessage = async (e) => {
        const { type, data } = e.data

        switch (type) {
          case 'compute':
            // Simple matrix operations
            const result = new Float32Array(data.size)
            for (let i = 0; i < data.size; i++) {
              result[i] = data.input[i] * data.weights[i % data.weights.length]
            }
            self.postMessage({ type: 'result', data: result })
            break
        }
      }
    `

    const blob = new Blob([workerCode], { type: 'application/javascript' })
    const worker = new Worker(URL.createObjectURL(blob))
    this.workers.push(worker)

    return worker
  }

  /**
   * Cleanup
   */
  destroy() {
    // Announce leave
    this.broadcast({ type: 'leave', nodeId: this.nodeId })

    // Cleanup timers
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer)
    if (this.cleanupTimer) clearInterval(this.cleanupTimer)

    // Close channel
    if (this.channel) this.channel.close()

    // Terminate workers
    for (const worker of this.workers) {
      worker.terminate()
    }

    this.ready = false
  }
}

export default DistributedGPU
