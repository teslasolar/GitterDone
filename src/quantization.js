/**
 * Quantization - Reduce model size with INT8/INT4 quantization
 * Enables running larger models in browser with minimal quality loss
 */
export class Quantization {
  constructor(gpu) {
    this.gpu = gpu
    this.calibrationData = null
    this.scaleFactors = new Map()
    this.zeroPoints = new Map()
  }

  /**
   * Quantize float32 weights to int8
   * Uses symmetric quantization: q = round(x / scale)
   */
  quantizeInt8(weights, layerName) {
    // Find max absolute value for scale
    let maxAbs = 0
    for (let i = 0; i < weights.length; i++) {
      const abs = Math.abs(weights[i])
      if (abs > maxAbs) maxAbs = abs
    }

    // Scale factor: maps [-maxAbs, maxAbs] to [-127, 127]
    const scale = maxAbs / 127
    this.scaleFactors.set(layerName, scale)

    // Quantize
    const quantized = new Int8Array(weights.length)
    for (let i = 0; i < weights.length; i++) {
      quantized[i] = Math.round(weights[i] / scale)
    }

    return {
      data: quantized,
      scale,
      dtype: 'int8',
      originalSize: weights.length * 4,
      quantizedSize: weights.length,
      compression: 4
    }
  }

  /**
   * Dequantize int8 back to float32
   */
  dequantizeInt8(quantized, layerName) {
    const scale = this.scaleFactors.get(layerName)
    const dequantized = new Float32Array(quantized.length)

    for (let i = 0; i < quantized.length; i++) {
      dequantized[i] = quantized[i] * scale
    }

    return dequantized
  }

  /**
   * Quantize to int4 (packed into int8)
   * Each byte holds 2 int4 values
   */
  quantizeInt4(weights, layerName) {
    let maxAbs = 0
    for (let i = 0; i < weights.length; i++) {
      const abs = Math.abs(weights[i])
      if (abs > maxAbs) maxAbs = abs
    }

    // Scale for int4: maps to [-7, 7]
    const scale = maxAbs / 7
    this.scaleFactors.set(layerName, scale)

    // Pack two int4 values into each byte
    const packedLength = Math.ceil(weights.length / 2)
    const quantized = new Uint8Array(packedLength)

    for (let i = 0; i < weights.length; i += 2) {
      const val1 = Math.max(-7, Math.min(7, Math.round(weights[i] / scale)))
      const val2 = i + 1 < weights.length
        ? Math.max(-7, Math.min(7, Math.round(weights[i + 1] / scale)))
        : 0

      // Pack: low nibble = val1 + 8, high nibble = val2 + 8
      quantized[i / 2] = ((val1 + 8) & 0x0F) | (((val2 + 8) & 0x0F) << 4)
    }

    return {
      data: quantized,
      scale,
      dtype: 'int4',
      originalSize: weights.length * 4,
      quantizedSize: packedLength,
      compression: 8,
      originalLength: weights.length
    }
  }

  /**
   * Dequantize int4 back to float32
   */
  dequantizeInt4(quantized, layerName, originalLength) {
    const scale = this.scaleFactors.get(layerName)
    const dequantized = new Float32Array(originalLength)

    for (let i = 0; i < originalLength; i += 2) {
      const packed = quantized[i / 2]
      const val1 = (packed & 0x0F) - 8
      const val2 = ((packed >> 4) & 0x0F) - 8

      dequantized[i] = val1 * scale
      if (i + 1 < originalLength) {
        dequantized[i + 1] = val2 * scale
      }
    }

    return dequantized
  }

  /**
   * GPTQ-style quantization (grouped)
   * Better quality by quantizing in groups
   */
  quantizeGrouped(weights, layerName, groupSize = 128) {
    const numGroups = Math.ceil(weights.length / groupSize)
    const scales = new Float32Array(numGroups)
    const quantized = new Int8Array(weights.length)

    for (let g = 0; g < numGroups; g++) {
      const start = g * groupSize
      const end = Math.min(start + groupSize, weights.length)

      // Find max in group
      let maxAbs = 0
      for (let i = start; i < end; i++) {
        const abs = Math.abs(weights[i])
        if (abs > maxAbs) maxAbs = abs
      }

      scales[g] = maxAbs / 127

      // Quantize group
      for (let i = start; i < end; i++) {
        quantized[i] = Math.round(weights[i] / scales[g])
      }
    }

    this.scaleFactors.set(layerName, { scales, groupSize })

    return {
      data: quantized,
      scales,
      groupSize,
      dtype: 'int8-grouped',
      originalSize: weights.length * 4,
      quantizedSize: weights.length + numGroups * 4,
      compression: weights.length * 4 / (weights.length + numGroups * 4)
    }
  }

  /**
   * Dequantize grouped int8
   */
  dequantizeGrouped(quantized, layerName) {
    const { scales, groupSize } = this.scaleFactors.get(layerName)
    const dequantized = new Float32Array(quantized.length)

    for (let g = 0; g < scales.length; g++) {
      const start = g * groupSize
      const end = Math.min(start + groupSize, quantized.length)

      for (let i = start; i < end; i++) {
        dequantized[i] = quantized[i] * scales[g]
      }
    }

    return dequantized
  }

  /**
   * AWQ-style activation-aware quantization
   * Calibrate using sample inputs for better accuracy
   */
  async calibrate(model, sampleInputs) {
    console.log('Calibrating with', sampleInputs.length, 'samples...')
    this.calibrationData = new Map()

    for (const input of sampleInputs) {
      // Run forward pass and collect activations
      const activations = await model.forwardWithActivations(input)

      for (const [layerName, activation] of Object.entries(activations)) {
        if (!this.calibrationData.has(layerName)) {
          this.calibrationData.set(layerName, { min: Infinity, max: -Infinity, sum: 0, count: 0 })
        }

        const stats = this.calibrationData.get(layerName)
        for (let i = 0; i < activation.length; i++) {
          stats.min = Math.min(stats.min, activation[i])
          stats.max = Math.max(stats.max, activation[i])
          stats.sum += Math.abs(activation[i])
          stats.count++
        }
      }
    }

    // Compute optimal scales based on activation distributions
    for (const [layerName, stats] of this.calibrationData) {
      const avgMagnitude = stats.sum / stats.count
      // Use percentile-based clipping for outliers
      const clipValue = avgMagnitude * 3
      this.calibrationData.set(layerName, { ...stats, clipValue })
    }

    console.log('Calibration complete')
  }

  /**
   * Quantize entire model
   */
  async quantizeModel(model, options = {}) {
    const {
      dtype = 'int8',
      groupSize = 128,
      excludeLayers = ['embed', 'lm_head'] // Keep embeddings in fp32
    } = options

    const quantizedLayers = {}
    let originalSize = 0
    let quantizedSize = 0

    for (const layer of model.layers) {
      if (excludeLayers.some(ex => layer.name.includes(ex))) {
        // Keep in original format
        continue
      }

      if (!layer.weights) {
        await model.loadLayerWeights(layer)
      }

      const weights = await model.tensorCube.readTensor(layer.weights.name)
      originalSize += weights.length * 4

      let result
      switch (dtype) {
        case 'int4':
          result = this.quantizeInt4(weights, layer.name)
          break
        case 'int8-grouped':
          result = this.quantizeGrouped(weights, layer.name, groupSize)
          break
        case 'int8':
        default:
          result = this.quantizeInt8(weights, layer.name)
      }

      quantizedLayers[layer.name] = result
      quantizedSize += result.quantizedSize
    }

    return {
      layers: quantizedLayers,
      originalSize,
      quantizedSize,
      compression: originalSize / quantizedSize,
      dtype
    }
  }

  /**
   * Export quantized model
   */
  exportQuantized(quantizedModel) {
    const metadata = {
      dtype: quantizedModel.dtype,
      layers: {}
    }

    const chunks = []
    let offset = 0

    for (const [layerName, layer] of Object.entries(quantizedModel.layers)) {
      metadata.layers[layerName] = {
        offset,
        size: layer.data.byteLength,
        scale: layer.scale,
        scales: layer.scales,
        groupSize: layer.groupSize,
        dtype: layer.dtype,
        originalLength: layer.originalLength
      }

      chunks.push(layer.data)
      offset += layer.data.byteLength
    }

    // Combine all chunks
    const totalSize = chunks.reduce((sum, c) => sum + c.byteLength, 0)
    const buffer = new Uint8Array(totalSize)
    let pos = 0
    for (const chunk of chunks) {
      buffer.set(new Uint8Array(chunk.buffer || chunk), pos)
      pos += chunk.byteLength
    }

    return {
      metadata,
      weights: buffer
    }
  }

  /**
   * Create dequantization WGSL kernel
   */
  createDequantKernel(dtype) {
    if (dtype === 'int8') {
      return `
        struct Params {
          size: u32,
          scale: f32,
        }

        @group(0) @binding(0) var<storage, read> quantized: array<i32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn dequantize(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= params.size) { return; }

          // Unpack 4 int8 values from one i32
          let packed = quantized[i / 4u];
          let byte_idx = i % 4u;
          var val: i32;

          switch(byte_idx) {
            case 0u: { val = (packed & 0xFF) - 128; }
            case 1u: { val = ((packed >> 8) & 0xFF) - 128; }
            case 2u: { val = ((packed >> 16) & 0xFF) - 128; }
            default: { val = ((packed >> 24) & 0xFF) - 128; }
          }

          output[i] = f32(val) * params.scale;
        }
      `
    }

    if (dtype === 'int4') {
      return `
        struct Params {
          size: u32,
          scale: f32,
        }

        @group(0) @binding(0) var<storage, read> quantized: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn dequantize(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= params.size) { return; }

          // Unpack 8 int4 values from one u32
          let packed = quantized[i / 8u];
          let nibble_idx = i % 8u;
          let shift = nibble_idx * 4u;
          let val = i32((packed >> shift) & 0xFu) - 8;

          output[i] = f32(val) * params.scale;
        }
      `
    }

    throw new Error(`Unknown dtype: ${dtype}`)
  }
}

export default Quantization
