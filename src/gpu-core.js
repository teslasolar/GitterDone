/**
 * GPUCore - WebGPU initialization and compute kernel management
 * Provides direct GPU access in the browser for ML inference
 */
export class GPUCore {
  constructor() {
    this.gpu = null
    this.device = null
    this.queue = null
    this.pipelines = new Map()
    this.bindGroupLayouts = new Map()
    this.initialized = false
  }

  /**
   * Initialize WebGPU adapter and device
   */
  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported! Use Chrome 113+ or Edge 113+')
    }

    this.gpu = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    })

    if (!this.gpu) {
      throw new Error('No WebGPU adapter found')
    }

    // Request device with maximum limits
    this.device = await this.gpu.requestDevice({
      requiredLimits: {
        maxBufferSize: this.gpu.limits.maxBufferSize,
        maxStorageBufferBindingSize: this.gpu.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: this.gpu.limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: this.gpu.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: this.gpu.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: this.gpu.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: this.gpu.limits.maxComputeWorkgroupSizeZ
      }
    })

    this.queue = this.device.queue
    this.initialized = true

    // Setup error handling
    this.device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message)
      this.initialized = false
    })

    return this
  }

  /**
   * Get GPU adapter info
   */
  getInfo() {
    if (!this.gpu) return null
    return {
      vendor: this.gpu.info?.vendor || 'Unknown',
      architecture: this.gpu.info?.architecture || 'Unknown',
      device: this.gpu.info?.device || 'Unknown',
      description: this.gpu.info?.description || 'Unknown',
      limits: {
        maxBufferSize: this.device.limits.maxBufferSize,
        maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: this.device.limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: this.device.limits.maxComputeInvocationsPerWorkgroup
      }
    }
  }

  /**
   * Create a compute pipeline from WGSL shader code
   */
  createKernel(name, shaderCode, entryPoint = 'main') {
    const shaderModule = this.device.createShaderModule({
      label: `${name}-shader`,
      code: shaderCode
    })

    const pipeline = this.device.createComputePipeline({
      label: `${name}-pipeline`,
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: entryPoint
      }
    })

    this.pipelines.set(name, pipeline)
    return pipeline
  }

  /**
   * Create a GPU buffer
   */
  createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage,
      mappedAtCreation: true
    })

    new (data.constructor)(buffer.getMappedRange()).set(data)
    buffer.unmap()

    return buffer
  }

  /**
   * Create an empty buffer of specified size
   */
  createEmptyBuffer(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    return this.device.createBuffer({
      size: size,
      usage: usage
    })
  }

  /**
   * Read data back from GPU buffer
   */
  async readBuffer(buffer, size) {
    const stagingBuffer = this.device.createBuffer({
      size: size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    const commandEncoder = this.device.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size)
    this.queue.submit([commandEncoder.finish()])

    await stagingBuffer.mapAsync(GPUMapMode.READ)
    const result = new Float32Array(stagingBuffer.getMappedRange().slice(0))
    stagingBuffer.unmap()
    stagingBuffer.destroy()

    return result
  }

  /**
   * Execute a compute kernel
   */
  async compute(kernelName, bindGroup, workgroupsX, workgroupsY = 1, workgroupsZ = 1) {
    const pipeline = this.pipelines.get(kernelName)
    if (!pipeline) {
      throw new Error(`Kernel '${kernelName}' not found`)
    }

    const commandEncoder = this.device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()

    passEncoder.setPipeline(pipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ)
    passEncoder.end()

    this.queue.submit([commandEncoder.finish()])
    await this.device.queue.onSubmittedWorkDone()
  }

  /**
   * Create bind group for kernel execution
   */
  createBindGroup(kernelName, buffers) {
    const pipeline = this.pipelines.get(kernelName)
    if (!pipeline) {
      throw new Error(`Kernel '${kernelName}' not found`)
    }

    const entries = buffers.map((buffer, index) => ({
      binding: index,
      resource: { buffer }
    }))

    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    })
  }

  /**
   * Destroy resources
   */
  destroy() {
    if (this.device) {
      this.device.destroy()
    }
    this.pipelines.clear()
    this.initialized = false
  }
}

export default GPUCore
