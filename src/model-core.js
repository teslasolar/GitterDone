/**
 * ModelCore - Universal Model Runner
 * Supports ONNX, custom binary formats, and runs inference on WebGPU
 */
import { ShardArray } from './shard-array.js'
import { TensorCube } from './tensor-cube.js'

export class ModelCore {
  constructor(gpu) {
    this.gpu = gpu
    this.tensorCube = new TensorCube(gpu)
    this.weights = null
    this.config = null
    this.layers = []
    this.graph = null
    this.ready = false
  }

  /**
   * Initialize the model from a URL
   */
  async init(modelUrl) {
    this.weights = new ShardArray(modelUrl)
    await this.weights.initCache()
    await this.tensorCube.initShaders()

    // Load model config
    this.config = await this.weights.loadMetadata()
    this.buildGraph()
    this.ready = true

    return this
  }

  /**
   * Build computational graph from config
   */
  buildGraph() {
    this.layers = []

    for (const layerConfig of this.config.layers || []) {
      const layer = {
        name: layerConfig.name,
        type: layerConfig.type,
        params: layerConfig.params || {},
        weightOffset: layerConfig.weight_offset,
        weightSize: layerConfig.weight_size,
        biasOffset: layerConfig.bias_offset,
        biasSize: layerConfig.bias_size,
        inputShape: layerConfig.input_shape,
        outputShape: layerConfig.output_shape,
        weights: null,
        bias: null
      }
      this.layers.push(layer)
    }
  }

  /**
   * Load weights for a specific layer
   */
  async loadLayerWeights(layer) {
    if (layer.weights) return // Already loaded

    if (layer.weightOffset !== undefined && layer.weightSize > 0) {
      const weightData = await this.weights.loadRange(
        layer.weightOffset,
        layer.weightOffset + layer.weightSize
      )
      layer.weights = this.tensorCube.createTensor(
        `${layer.name}_weight`,
        weightData,
        layer.params.weight_shape || [layer.weightSize / 4]
      )
    }

    if (layer.biasOffset !== undefined && layer.biasSize > 0) {
      const biasData = await this.weights.loadRange(
        layer.biasOffset,
        layer.biasOffset + layer.biasSize
      )
      layer.bias = this.tensorCube.createTensor(
        `${layer.name}_bias`,
        biasData,
        layer.params.bias_shape || [layer.biasSize / 4]
      )
    }
  }

  /**
   * Forward pass through the model
   */
  async forward(input) {
    if (!this.ready) {
      throw new Error('Model not initialized. Call init() first.')
    }

    // Create input tensor
    let x = this.tensorCube.createTensor('input', input, [input.length])

    // Process each layer
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]
      await this.loadLayerWeights(layer)

      x = await this.computeLayer(layer, x, i)
    }

    // Read output
    return await this.tensorCube.readTensor(x.name)
  }

  /**
   * Compute a single layer
   */
  async computeLayer(layer, input, index) {
    const outputName = `layer_${index}_out`

    switch (layer.type) {
      case 'linear':
      case 'dense':
        return await this.computeLinear(layer, input, outputName)

      case 'embedding':
        return await this.computeEmbedding(layer, input, outputName)

      case 'layernorm':
      case 'layer_norm':
        return await this.computeLayerNorm(layer, input, outputName)

      case 'attention':
        return await this.computeAttention(layer, input, outputName)

      case 'relu':
        return await this.tensorCube.relu(input.name, outputName)

      case 'gelu':
        return await this.tensorCube.gelu(input.name, outputName)

      case 'softmax':
        return await this.tensorCube.softmax(input.name, outputName, layer.params.temperature || 1.0)

      case 'residual':
        return await this.computeResidual(layer, input, outputName)

      default:
        console.warn(`Unknown layer type: ${layer.type}, passing through`)
        return input
    }
  }

  /**
   * Compute linear/dense layer: y = xW + b
   */
  async computeLinear(layer, input, outputName) {
    const inFeatures = layer.params.in_features || layer.inputShape?.[1] || input.size
    const outFeatures = layer.params.out_features || layer.outputShape?.[1]

    // Matrix multiply
    const tempName = `${outputName}_matmul`
    await this.tensorCube.matmul(
      input.name,
      layer.weights.name,
      tempName,
      1, // batch size
      outFeatures,
      inFeatures
    )

    // Add bias if present
    if (layer.bias) {
      return await this.tensorCube.add(tempName, layer.bias.name, outputName)
    }

    return this.tensorCube.getTensor(tempName)
  }

  /**
   * Compute embedding lookup
   */
  async computeEmbedding(layer, input, outputName) {
    // For embeddings, input is token IDs
    const inputData = await this.tensorCube.readTensor(input.name)
    const vocabSize = layer.params.vocab_size
    const embedDim = layer.params.embed_dim

    // Read embedding weights
    const embedWeights = await this.tensorCube.readTensor(layer.weights.name)

    // Lookup embeddings for each token
    const seqLen = inputData.length
    const output = new Float32Array(seqLen * embedDim)

    for (let i = 0; i < seqLen; i++) {
      const tokenId = Math.floor(inputData[i])
      const start = tokenId * embedDim
      for (let j = 0; j < embedDim; j++) {
        output[i * embedDim + j] = embedWeights[start + j]
      }
    }

    return this.tensorCube.createTensor(outputName, output, [seqLen, embedDim])
  }

  /**
   * Compute layer normalization
   */
  async computeLayerNorm(layer, input, outputName) {
    const inputData = await this.tensorCube.readTensor(input.name)
    const epsilon = layer.params.epsilon || 1e-5

    // Compute mean
    let mean = 0
    for (let i = 0; i < inputData.length; i++) {
      mean += inputData[i]
    }
    mean /= inputData.length

    // Compute variance
    let variance = 0
    for (let i = 0; i < inputData.length; i++) {
      variance += (inputData[i] - mean) ** 2
    }
    variance /= inputData.length

    // Normalize
    const output = new Float32Array(inputData.length)
    const gamma = layer.weights ? await this.tensorCube.readTensor(layer.weights.name) : null
    const beta = layer.bias ? await this.tensorCube.readTensor(layer.bias.name) : null

    for (let i = 0; i < inputData.length; i++) {
      let normalized = (inputData[i] - mean) / Math.sqrt(variance + epsilon)
      if (gamma) normalized *= gamma[i % gamma.length]
      if (beta) normalized += beta[i % beta.length]
      output[i] = normalized
    }

    return this.tensorCube.createTensor(outputName, output, input.shape)
  }

  /**
   * Compute self-attention
   */
  async computeAttention(layer, input, outputName) {
    const inputData = await this.tensorCube.readTensor(input.name)
    const numHeads = layer.params.num_heads || 1
    const headDim = layer.params.head_dim || inputData.length / numHeads
    const seqLen = layer.params.seq_len || 1

    // Simplified attention: Q, K, V projections
    // For full implementation, this would use separate weight matrices
    const embedWeights = await this.tensorCube.readTensor(layer.weights.name)

    // Compute Q, K, V (simplified: same projection for demo)
    const qkv = new Float32Array(inputData.length * 3)
    for (let i = 0; i < inputData.length; i++) {
      // Simple linear transformation
      qkv[i] = inputData[i] * embedWeights[i % embedWeights.length]
      qkv[i + inputData.length] = inputData[i] * embedWeights[(i + 1) % embedWeights.length]
      qkv[i + inputData.length * 2] = inputData[i] * embedWeights[(i + 2) % embedWeights.length]
    }

    // Compute attention scores
    const scores = new Float32Array(seqLen * seqLen)
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        let score = 0
        for (let k = 0; k < headDim; k++) {
          score += qkv[i * headDim + k] * qkv[inputData.length + j * headDim + k]
        }
        scores[i * seqLen + j] = score / Math.sqrt(headDim)
      }
    }

    // Softmax on scores
    for (let i = 0; i < seqLen; i++) {
      let max = -Infinity
      for (let j = 0; j < seqLen; j++) {
        if (scores[i * seqLen + j] > max) max = scores[i * seqLen + j]
      }
      let sum = 0
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] = Math.exp(scores[i * seqLen + j] - max)
        sum += scores[i * seqLen + j]
      }
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] /= sum
      }
    }

    // Apply attention to values
    const output = new Float32Array(inputData.length)
    for (let i = 0; i < seqLen; i++) {
      for (let k = 0; k < headDim; k++) {
        let sum = 0
        for (let j = 0; j < seqLen; j++) {
          sum += scores[i * seqLen + j] * qkv[inputData.length * 2 + j * headDim + k]
        }
        output[i * headDim + k] = sum
      }
    }

    return this.tensorCube.createTensor(outputName, output, input.shape)
  }

  /**
   * Compute residual connection
   */
  async computeResidual(layer, input, outputName) {
    // Residual assumes previous layer output is stored
    const residualName = layer.params.residual_from || 'input'
    return await this.tensorCube.add(input.name, residualName, outputName)
  }

  /**
   * Get model info
   */
  getInfo() {
    return {
      name: this.config?.name || 'Unknown',
      version: this.config?.version || '1.0',
      layers: this.layers.length,
      parameters: this.config?.parameters || 0,
      vocabSize: this.config?.vocab_size || 0,
      embedDim: this.config?.embed_dim || 0,
      shards: this.config?.shards || 1
    }
  }

  /**
   * Cleanup resources
   */
  destroy() {
    this.tensorCube.destroy()
    this.weights?.clearMemoryCache()
    this.layers = []
    this.ready = false
  }
}

export default ModelCore
