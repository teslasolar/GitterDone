/**
 * Pipeline - Streaming Inference for Text Generation
 * Generates tokens as they compute, with real-time output
 */
export class Pipeline {
  constructor(model) {
    this.model = model
    this.tokenizer = null
    this.buffer = []
    this.streaming = false
    this.stopped = false
    this.kvCache = null // Key-value cache for faster generation
  }

  /**
   * Initialize tokenizer from model config
   */
  async initTokenizer(tokenizerUrl = null) {
    const url = tokenizerUrl || `${this.model.weights.baseUrl}/tokenizer.json`

    try {
      const response = await fetch(url)
      if (!response.ok) {
        console.warn('No tokenizer.json found, using default char tokenizer')
        this.tokenizer = this.createCharTokenizer()
        return
      }
      const config = await response.json()
      this.tokenizer = this.buildTokenizer(config)
    } catch (error) {
      console.warn('Failed to load tokenizer, using default:', error)
      this.tokenizer = this.createCharTokenizer()
    }
  }

  /**
   * Build tokenizer from config
   */
  buildTokenizer(config) {
    const vocab = config.vocab || {}
    const merges = config.merges || []
    const specialTokens = config.special_tokens || {}

    // Build vocab maps
    const tokenToId = new Map(Object.entries(vocab))
    const idToToken = new Map(Object.entries(vocab).map(([k, v]) => [v, k]))

    // BPE merges
    const bpeMerges = new Map()
    merges.forEach((merge, i) => {
      bpeMerges.set(merge, i)
    })

    return {
      vocab,
      tokenToId,
      idToToken,
      bpeMerges,
      specialTokens,
      padTokenId: specialTokens.pad_token_id ?? 0,
      eosTokenId: specialTokens.eos_token_id ?? 2,
      bosTokenId: specialTokens.bos_token_id ?? 1,
      unkTokenId: specialTokens.unk_token_id ?? 3,

      encode: (text) => this.bpeEncode(text, tokenToId, bpeMerges),
      decode: (ids) => this.bpeDecode(ids, idToToken)
    }
  }

  /**
   * Create simple character-level tokenizer
   */
  createCharTokenizer() {
    const vocab = {}
    const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\'"-:;()[]{}@#$%^&*+=<>/\\|`~\n\t'

    chars.split('').forEach((char, i) => {
      vocab[char] = i + 4 // Reserve 0-3 for special tokens
    })

    const tokenToId = new Map(Object.entries(vocab))
    const idToToken = new Map(Object.entries(vocab).map(([k, v]) => [v, k]))

    return {
      vocab,
      tokenToId,
      idToToken,
      padTokenId: 0,
      eosTokenId: 1,
      bosTokenId: 2,
      unkTokenId: 3,

      encode: (text) => {
        const ids = [2] // BOS
        for (const char of text) {
          ids.push(tokenToId.get(char) ?? 3) // UNK for unknown
        }
        return ids
      },

      decode: (ids) => {
        return ids
          .filter(id => id > 3) // Skip special tokens
          .map(id => idToToken.get(id) ?? '')
          .join('')
      }
    }
  }

  /**
   * BPE encode text
   */
  bpeEncode(text, tokenToId, bpeMerges) {
    // Simplified BPE encoding
    const tokens = []

    // Pre-tokenize by splitting on whitespace
    const words = text.split(/(\s+)/)

    for (const word of words) {
      if (!word) continue

      // Split word into characters
      let chars = word.split('').map(c => c)

      // Apply BPE merges
      while (chars.length > 1) {
        let minRank = Infinity
        let minPair = null
        let minIndex = -1

        // Find best merge
        for (let i = 0; i < chars.length - 1; i++) {
          const pair = `${chars[i]} ${chars[i + 1]}`
          const rank = bpeMerges.get(pair)
          if (rank !== undefined && rank < minRank) {
            minRank = rank
            minPair = pair
            minIndex = i
          }
        }

        if (minPair === null) break

        // Apply merge
        const merged = chars[minIndex] + chars[minIndex + 1]
        chars = [...chars.slice(0, minIndex), merged, ...chars.slice(minIndex + 2)]
      }

      // Convert to token IDs
      for (const token of chars) {
        const id = tokenToId.get(token)
        tokens.push(id ?? 3) // UNK for unknown
      }
    }

    return tokens
  }

  /**
   * BPE decode token IDs
   */
  bpeDecode(ids, idToToken) {
    return ids
      .map(id => idToToken.get(id) ?? '')
      .join('')
      .replace(/Ġ/g, ' ') // Handle space prefix
      .replace(/Ċ/g, '\n') // Handle newline
  }

  /**
   * Tokenize text
   */
  tokenize(text) {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized. Call initTokenizer() first.')
    }
    return this.tokenizer.encode(text)
  }

  /**
   * Decode tokens to text
   */
  decode(tokens) {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized. Call initTokenizer() first.')
    }
    return this.tokenizer.decode(tokens)
  }

  /**
   * Sample next token from logits
   */
  sample(logits, temperature = 0.8, topK = 40, topP = 0.9) {
    // Apply temperature
    const scaled = new Float32Array(logits.length)
    for (let i = 0; i < logits.length; i++) {
      scaled[i] = logits[i] / temperature
    }

    // Get sorted indices
    const indices = Array.from({ length: logits.length }, (_, i) => i)
    indices.sort((a, b) => scaled[b] - scaled[a])

    // Apply top-k
    const topKIndices = indices.slice(0, topK)

    // Compute softmax on top-k
    let max = scaled[topKIndices[0]]
    let sum = 0
    const probs = new Float32Array(topKIndices.length)

    for (let i = 0; i < topKIndices.length; i++) {
      probs[i] = Math.exp(scaled[topKIndices[i]] - max)
      sum += probs[i]
    }

    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sum
    }

    // Apply top-p (nucleus sampling)
    let cumSum = 0
    let cutoff = probs.length
    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i]
      if (cumSum >= topP) {
        cutoff = i + 1
        break
      }
    }

    // Renormalize
    const finalProbs = probs.slice(0, cutoff)
    const finalSum = finalProbs.reduce((a, b) => a + b, 0)
    for (let i = 0; i < finalProbs.length; i++) {
      finalProbs[i] /= finalSum
    }

    // Sample from distribution
    const r = Math.random()
    let cumulative = 0
    for (let i = 0; i < finalProbs.length; i++) {
      cumulative += finalProbs[i]
      if (r < cumulative) {
        return topKIndices[i]
      }
    }

    return topKIndices[0]
  }

  /**
   * Generate tokens with streaming output
   */
  async *generate(prompt, options = {}) {
    const {
      maxTokens = 100,
      temperature = 0.8,
      topK = 40,
      topP = 0.9,
      stopTokens = null,
      onToken = null
    } = options

    if (!this.tokenizer) {
      await this.initTokenizer()
    }

    // Tokenize input
    let tokens = this.tokenize(prompt)
    this.streaming = true
    this.stopped = false

    const generatedTokens = []
    const startTime = performance.now()

    for (let i = 0; i < maxTokens && this.streaming && !this.stopped; i++) {
      // Create input tensor from tokens
      const input = new Float32Array(tokens)

      // Forward pass
      const logits = await this.model.forward(input)

      // Sample next token
      const nextToken = this.sample(logits, temperature, topK, topP)

      // Check for stop conditions
      if (nextToken === this.tokenizer.eosTokenId) {
        break
      }

      if (stopTokens && stopTokens.includes(nextToken)) {
        break
      }

      // Add to token sequence
      tokens.push(nextToken)
      generatedTokens.push(nextToken)

      // Decode and yield
      const text = this.decode([nextToken])

      if (onToken) {
        onToken({
          token: nextToken,
          text,
          index: i,
          elapsed: performance.now() - startTime
        })
      }

      yield text
    }

    this.streaming = false
  }

  /**
   * Generate complete response (non-streaming)
   */
  async generateSync(prompt, options = {}) {
    const tokens = []
    for await (const token of this.generate(prompt, options)) {
      tokens.push(token)
    }
    return tokens.join('')
  }

  /**
   * Stop generation
   */
  stop() {
    this.stopped = true
    this.streaming = false
  }

  /**
   * Check if currently generating
   */
  isGenerating() {
    return this.streaming
  }

  /**
   * Clear KV cache (for new conversations)
   */
  clearCache() {
    this.kvCache = null
    this.buffer = []
  }
}

export default Pipeline
