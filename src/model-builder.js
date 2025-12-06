/**
 * ModelBuilder - Visual model architecture builder
 * Allows users to design custom LLM architectures without code
 */
export class ModelBuilder {
  constructor() {
    this.config = {
      name: 'MyModel',
      version: '1.0.0',
      architecture: 'transformer',
      vocab_size: 32000,
      embed_dim: 256,
      hidden_dim: 512,
      num_heads: 8,
      num_layers: 4,
      max_seq_len: 512,
      dtype: 'float32'
    }

    this.layers = []
    this.presets = this.getPresets()
  }

  /**
   * Pre-built architecture templates
   */
  getPresets() {
    return {
      // Nano models (< 1MB) - Great for learning/demos
      'nano-gpt': {
        name: 'NanoGPT',
        description: 'Tiny GPT for learning (~500KB)',
        config: {
          vocab_size: 256,
          embed_dim: 64,
          hidden_dim: 128,
          num_heads: 4,
          num_layers: 2,
          max_seq_len: 128
        },
        estimatedSize: '500KB',
        useCase: 'Learning, demos, edge devices'
      },

      // Micro models (1-10MB) - Mobile/browser friendly
      'micro-llm': {
        name: 'MicroLLM',
        description: 'Small but capable (~5MB)',
        config: {
          vocab_size: 8000,
          embed_dim: 256,
          hidden_dim: 512,
          num_heads: 8,
          num_layers: 4,
          max_seq_len: 256
        },
        estimatedSize: '5MB',
        useCase: 'Mobile apps, browser, IoT'
      },

      // Small models (10-100MB) - Good balance
      'small-llm': {
        name: 'SmallLLM',
        description: 'Balanced performance (~50MB)',
        config: {
          vocab_size: 32000,
          embed_dim: 512,
          hidden_dim: 1024,
          num_heads: 8,
          num_layers: 6,
          max_seq_len: 512
        },
        estimatedSize: '50MB',
        useCase: 'General purpose, chatbots'
      },

      // Medium models (100MB-1GB)
      'medium-llm': {
        name: 'MediumLLM',
        description: 'Strong performance (~300MB)',
        config: {
          vocab_size: 32000,
          embed_dim: 768,
          hidden_dim: 2048,
          num_heads: 12,
          num_layers: 12,
          max_seq_len: 1024
        },
        estimatedSize: '300MB',
        useCase: 'Production apps, quality output'
      },

      // Specialized architectures
      'code-model': {
        name: 'CodeModel',
        description: 'Optimized for code generation',
        config: {
          vocab_size: 50000, // Larger vocab for code tokens
          embed_dim: 512,
          hidden_dim: 2048,
          num_heads: 8,
          num_layers: 8,
          max_seq_len: 2048, // Longer context for code
          rope_scaling: true
        },
        estimatedSize: '150MB',
        useCase: 'Code completion, generation'
      },

      'chat-model': {
        name: 'ChatModel',
        description: 'Optimized for conversations',
        config: {
          vocab_size: 32000,
          embed_dim: 512,
          hidden_dim: 1536,
          num_heads: 8,
          num_layers: 8,
          max_seq_len: 4096,
          use_sliding_window: true,
          window_size: 1024
        },
        estimatedSize: '100MB',
        useCase: 'Chatbots, assistants'
      }
    }
  }

  /**
   * Apply a preset configuration
   */
  applyPreset(presetName) {
    const preset = this.presets[presetName]
    if (!preset) throw new Error(`Unknown preset: ${presetName}`)

    this.config = {
      ...this.config,
      name: preset.name,
      ...preset.config
    }

    this.buildLayers()
    return this
  }

  /**
   * Set individual config values
   */
  setConfig(key, value) {
    this.config[key] = value
    return this
  }

  /**
   * Build layer configuration from high-level config
   */
  buildLayers() {
    this.layers = []
    let offset = 0

    // 1. Token Embedding
    const embedSize = this.config.vocab_size * this.config.embed_dim * 4
    this.layers.push({
      name: 'embed',
      type: 'embedding',
      params: {
        vocab_size: this.config.vocab_size,
        embed_dim: this.config.embed_dim
      },
      weight_offset: offset,
      weight_size: embedSize,
      input_shape: [this.config.max_seq_len],
      output_shape: [this.config.max_seq_len, this.config.embed_dim]
    })
    offset += embedSize

    // 2. Transformer layers
    for (let i = 0; i < this.config.num_layers; i++) {
      // Pre-attention LayerNorm
      const lnSize = this.config.embed_dim * 4 * 2 // gamma + beta
      this.layers.push({
        name: `layer${i}_ln1`,
        type: 'layernorm',
        params: { epsilon: 1e-5 },
        weight_offset: offset,
        weight_size: lnSize / 2,
        bias_offset: offset + lnSize / 2,
        bias_size: lnSize / 2,
        input_shape: [this.config.max_seq_len, this.config.embed_dim],
        output_shape: [this.config.max_seq_len, this.config.embed_dim]
      })
      offset += lnSize

      // Self-Attention (Q, K, V, O projections)
      const attnSize = this.config.embed_dim * this.config.embed_dim * 4 * 4 // 4 matrices
      this.layers.push({
        name: `layer${i}_attn`,
        type: 'attention',
        params: {
          num_heads: this.config.num_heads,
          head_dim: this.config.embed_dim / this.config.num_heads,
          seq_len: this.config.max_seq_len,
          causal: true
        },
        weight_offset: offset,
        weight_size: attnSize,
        input_shape: [this.config.max_seq_len, this.config.embed_dim],
        output_shape: [this.config.max_seq_len, this.config.embed_dim]
      })
      offset += attnSize

      // Residual connection (implicit)
      this.layers.push({
        name: `layer${i}_res1`,
        type: 'residual',
        params: { from: `layer${i}_ln1` },
        input_shape: [this.config.max_seq_len, this.config.embed_dim],
        output_shape: [this.config.max_seq_len, this.config.embed_dim]
      })

      // Post-attention LayerNorm
      this.layers.push({
        name: `layer${i}_ln2`,
        type: 'layernorm',
        params: { epsilon: 1e-5 },
        weight_offset: offset,
        weight_size: lnSize / 2,
        bias_offset: offset + lnSize / 2,
        bias_size: lnSize / 2,
        input_shape: [this.config.max_seq_len, this.config.embed_dim],
        output_shape: [this.config.max_seq_len, this.config.embed_dim]
      })
      offset += lnSize

      // FFN Up projection
      const ffnUpSize = this.config.embed_dim * this.config.hidden_dim * 4
      this.layers.push({
        name: `layer${i}_ffn_up`,
        type: 'linear',
        params: {
          in_features: this.config.embed_dim,
          out_features: this.config.hidden_dim
        },
        weight_offset: offset,
        weight_size: ffnUpSize,
        bias_offset: offset + ffnUpSize,
        bias_size: this.config.hidden_dim * 4,
        input_shape: [this.config.max_seq_len, this.config.embed_dim],
        output_shape: [this.config.max_seq_len, this.config.hidden_dim]
      })
      offset += ffnUpSize + this.config.hidden_dim * 4

      // Activation
      this.layers.push({
        name: `layer${i}_act`,
        type: 'gelu',
        params: {},
        input_shape: [this.config.max_seq_len, this.config.hidden_dim],
        output_shape: [this.config.max_seq_len, this.config.hidden_dim]
      })

      // FFN Down projection
      const ffnDownSize = this.config.hidden_dim * this.config.embed_dim * 4
      this.layers.push({
        name: `layer${i}_ffn_down`,
        type: 'linear',
        params: {
          in_features: this.config.hidden_dim,
          out_features: this.config.embed_dim
        },
        weight_offset: offset,
        weight_size: ffnDownSize,
        bias_offset: offset + ffnDownSize,
        bias_size: this.config.embed_dim * 4,
        input_shape: [this.config.max_seq_len, this.config.hidden_dim],
        output_shape: [this.config.max_seq_len, this.config.embed_dim]
      })
      offset += ffnDownSize + this.config.embed_dim * 4

      // Residual connection
      this.layers.push({
        name: `layer${i}_res2`,
        type: 'residual',
        params: { from: `layer${i}_ln2` },
        input_shape: [this.config.max_seq_len, this.config.embed_dim],
        output_shape: [this.config.max_seq_len, this.config.embed_dim]
      })
    }

    // 3. Final LayerNorm
    const finalLnSize = this.config.embed_dim * 4 * 2
    this.layers.push({
      name: 'final_ln',
      type: 'layernorm',
      params: { epsilon: 1e-5 },
      weight_offset: offset,
      weight_size: finalLnSize / 2,
      bias_offset: offset + finalLnSize / 2,
      bias_size: finalLnSize / 2,
      input_shape: [this.config.max_seq_len, this.config.embed_dim],
      output_shape: [this.config.max_seq_len, this.config.embed_dim]
    })
    offset += finalLnSize

    // 4. LM Head (output projection)
    const lmHeadSize = this.config.embed_dim * this.config.vocab_size * 4
    this.layers.push({
      name: 'lm_head',
      type: 'linear',
      params: {
        in_features: this.config.embed_dim,
        out_features: this.config.vocab_size
      },
      weight_offset: offset,
      weight_size: lmHeadSize,
      input_shape: [this.config.max_seq_len, this.config.embed_dim],
      output_shape: [this.config.max_seq_len, this.config.vocab_size]
    })
    offset += lmHeadSize

    // Store total size
    this.totalWeightSize = offset

    return this
  }

  /**
   * Estimate model size
   */
  estimateSize() {
    this.buildLayers()

    const bytesPerParam = this.config.dtype === 'float16' ? 2 : 4
    const totalParams = this.totalWeightSize / 4 // Stored as float32 during build

    return {
      parameters: totalParams,
      sizeBytes: totalParams * bytesPerParam,
      sizeMB: (totalParams * bytesPerParam) / (1024 * 1024),
      shards: Math.ceil((totalParams * bytesPerParam) / (50 * 1024 * 1024)),
      layers: this.layers.length
    }
  }

  /**
   * Generate complete config.json
   */
  toConfig() {
    this.buildLayers()
    const estimate = this.estimateSize()

    return {
      name: this.config.name,
      version: this.config.version,
      description: `Custom ${this.config.architecture} model`,
      architecture: this.config.architecture,
      parameters: estimate.parameters,
      vocab_size: this.config.vocab_size,
      embed_dim: this.config.embed_dim,
      hidden_dim: this.config.hidden_dim,
      num_heads: this.config.num_heads,
      num_layers: this.config.num_layers,
      max_seq_len: this.config.max_seq_len,
      shards: estimate.shards,
      shard_size: 50 * 1024 * 1024,
      dtype: this.config.dtype,
      layers: this.layers
    }
  }

  /**
   * Export to JSON string
   */
  toJSON() {
    return JSON.stringify(this.toConfig(), null, 2)
  }

  /**
   * Validate configuration
   */
  validate() {
    const errors = []

    if (this.config.embed_dim % this.config.num_heads !== 0) {
      errors.push(`embed_dim (${this.config.embed_dim}) must be divisible by num_heads (${this.config.num_heads})`)
    }

    if (this.config.vocab_size < 256) {
      errors.push(`vocab_size (${this.config.vocab_size}) should be at least 256`)
    }

    if (this.config.num_layers < 1) {
      errors.push('num_layers must be at least 1')
    }

    const estimate = this.estimateSize()
    if (estimate.sizeMB > 4000) {
      errors.push(`Model size (${estimate.sizeMB.toFixed(0)}MB) exceeds browser memory limits`)
    }

    return {
      valid: errors.length === 0,
      errors
    }
  }
}

export default ModelBuilder
