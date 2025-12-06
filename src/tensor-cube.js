/**
 * TensorCube - 3D Tensor Operations with WebGPU
 * Manages tensor buffers and GPU compute operations in a 3D space
 */
export class TensorCube {
  constructor(gpu, dimensions = [256, 256, 256]) {
    this.gpu = gpu
    this.dimensions = dimensions
    this.buffers = new Map() // coord key → GPU buffer
    this.tensors = new Map() // name → tensor metadata
    this.shaders = new Map() // operation → compiled kernel
    this.workgroupSize = 16
  }

  /**
   * Initialize standard compute shaders
   */
  async initShaders() {
    // Matrix multiplication shader
    this.gpu.createKernel('matmul', `
      struct Uniforms {
        M: u32,
        N: u32,
        K: u32,
      }
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: array<f32>;
      @group(0) @binding(3) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let row = id.x;
        let col = id.y;
        if (row >= uniforms.M || col >= uniforms.N) { return; }

        var sum = 0.0;
        for (var i = 0u; i < uniforms.K; i++) {
          sum += a[row * uniforms.K + i] * b[i * uniforms.N + col];
        }
        c[row * uniforms.N + col] = sum;
      }
    `)

    // Element-wise addition
    this.gpu.createKernel('add', `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        c[i] = a[i] + b[i];
      }
    `)

    // Element-wise multiply
    this.gpu.createKernel('mul', `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        c[i] = a[i] * b[i];
      }
    `)

    // ReLU activation
    this.gpu.createKernel('relu', `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        output[i] = max(0.0, input[i]);
      }
    `)

    // GELU activation (approximation)
    this.gpu.createKernel('gelu', `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        let x = input[i];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let c = 0.7978845608; // sqrt(2/pi)
        let inner = c * (x + 0.044715 * x * x * x);
        output[i] = 0.5 * x * (1.0 + tanh(inner));
      }
    `)

    // Softmax (requires two passes: max/exp, then normalize)
    this.gpu.createKernel('softmax_exp', `
      struct Params { size: u32, max_val: f32 }
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      @group(0) @binding(2) var<uniform> params: Params;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        if (i >= params.size) { return; }
        output[i] = exp(input[i] - params.max_val);
      }
    `)

    this.gpu.createKernel('softmax_norm', `
      struct Params { size: u32, sum: f32 }
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      @group(0) @binding(2) var<uniform> params: Params;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        if (i >= params.size) { return; }
        output[i] = input[i] / params.sum;
      }
    `)

    // Layer normalization
    this.gpu.createKernel('layernorm', `
      struct Params {
        size: u32,
        mean: f32,
        variance: f32,
        epsilon: f32,
      }
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> gamma: array<f32>;
      @group(0) @binding(2) var<storage, read> beta: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32>;
      @group(0) @binding(4) var<uniform> params: Params;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        if (i >= params.size) { return; }
        let normalized = (input[i] - params.mean) / sqrt(params.variance + params.epsilon);
        output[i] = gamma[i] * normalized + beta[i];
      }
    `)

    return this
  }

  /**
   * Create a tensor from data
   */
  createTensor(name, data, shape) {
    const buffer = this.gpu.createBuffer(data)
    const tensor = {
      name,
      buffer,
      shape,
      size: data.length,
      dtype: 'float32'
    }
    this.tensors.set(name, tensor)
    return tensor
  }

  /**
   * Create an empty tensor
   */
  createEmptyTensor(name, shape) {
    const size = shape.reduce((a, b) => a * b, 1)
    const buffer = this.gpu.createEmptyBuffer(size * 4) // float32 = 4 bytes
    const tensor = {
      name,
      buffer,
      shape,
      size,
      dtype: 'float32'
    }
    this.tensors.set(name, tensor)
    return tensor
  }

  /**
   * Get tensor by name
   */
  getTensor(name) {
    return this.tensors.get(name)
  }

  /**
   * Read tensor data back from GPU
   */
  async readTensor(name) {
    const tensor = this.tensors.get(name)
    if (!tensor) throw new Error(`Tensor '${name}' not found`)
    return await this.gpu.readBuffer(tensor.buffer, tensor.size * 4)
  }

  /**
   * Matrix multiplication: C = A @ B
   */
  async matmul(aName, bName, cName, M, N, K) {
    const a = this.getTensor(aName)
    const b = this.getTensor(bName)
    let c = this.getTensor(cName)

    if (!c) {
      c = this.createEmptyTensor(cName, [M, N])
    }

    // Create uniforms buffer
    const uniforms = new Uint32Array([M, N, K])
    const uniformBuffer = this.gpu.createBuffer(uniforms, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)

    const bindGroup = this.gpu.device.createBindGroup({
      layout: this.gpu.pipelines.get('matmul').getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: c.buffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    })

    const workgroupsX = Math.ceil(M / this.workgroupSize)
    const workgroupsY = Math.ceil(N / this.workgroupSize)

    await this.gpu.compute('matmul', bindGroup, workgroupsX, workgroupsY, 1)

    uniformBuffer.destroy()
    return c
  }

  /**
   * Element-wise addition: C = A + B
   */
  async add(aName, bName, cName) {
    const a = this.getTensor(aName)
    const b = this.getTensor(bName)
    let c = this.getTensor(cName)

    if (!c) {
      c = this.createEmptyTensor(cName, a.shape)
    }

    const bindGroup = this.gpu.createBindGroup('add', [a.buffer, b.buffer, c.buffer])
    const workgroups = Math.ceil(a.size / 256)

    await this.gpu.compute('add', bindGroup, workgroups)
    return c
  }

  /**
   * Element-wise multiply: C = A * B
   */
  async mul(aName, bName, cName) {
    const a = this.getTensor(aName)
    const b = this.getTensor(bName)
    let c = this.getTensor(cName)

    if (!c) {
      c = this.createEmptyTensor(cName, a.shape)
    }

    const bindGroup = this.gpu.createBindGroup('mul', [a.buffer, b.buffer, c.buffer])
    const workgroups = Math.ceil(a.size / 256)

    await this.gpu.compute('mul', bindGroup, workgroups)
    return c
  }

  /**
   * ReLU activation
   */
  async relu(inputName, outputName) {
    const input = this.getTensor(inputName)
    let output = this.getTensor(outputName)

    if (!output) {
      output = this.createEmptyTensor(outputName, input.shape)
    }

    const bindGroup = this.gpu.createBindGroup('relu', [input.buffer, output.buffer])
    const workgroups = Math.ceil(input.size / 256)

    await this.gpu.compute('relu', bindGroup, workgroups)
    return output
  }

  /**
   * GELU activation
   */
  async gelu(inputName, outputName) {
    const input = this.getTensor(inputName)
    let output = this.getTensor(outputName)

    if (!output) {
      output = this.createEmptyTensor(outputName, input.shape)
    }

    const bindGroup = this.gpu.createBindGroup('gelu', [input.buffer, output.buffer])
    const workgroups = Math.ceil(input.size / 256)

    await this.gpu.compute('gelu', bindGroup, workgroups)
    return output
  }

  /**
   * Softmax (CPU fallback for now, GPU version requires reduction)
   */
  async softmax(inputName, outputName, temperature = 1.0) {
    const input = this.getTensor(inputName)
    const data = await this.readTensor(inputName)

    // Apply temperature and compute max for numerical stability
    const scaled = new Float32Array(data.length)
    let max = -Infinity
    for (let i = 0; i < data.length; i++) {
      scaled[i] = data[i] / temperature
      if (scaled[i] > max) max = scaled[i]
    }

    // Compute exp and sum
    let sum = 0
    const exp = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) {
      exp[i] = Math.exp(scaled[i] - max)
      sum += exp[i]
    }

    // Normalize
    const result = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) {
      result[i] = exp[i] / sum
    }

    // Create output tensor
    let output = this.getTensor(outputName)
    if (output) {
      output.buffer.destroy()
    }
    return this.createTensor(outputName, result, input.shape)
  }

  /**
   * Get buffer at 3D coordinates
   */
  getBuffer(x, y, z) {
    const key = `${x},${y},${z}`
    return this.buffers.get(key)
  }

  /**
   * Set buffer at 3D coordinates
   */
  setBuffer(x, y, z, buffer) {
    const key = `${x},${y},${z}`
    this.buffers.set(key, buffer)
  }

  /**
   * Cleanup all tensors
   */
  destroy() {
    for (const tensor of this.tensors.values()) {
      tensor.buffer.destroy()
    }
    this.tensors.clear()
    this.buffers.clear()
  }
}

export default TensorCube
