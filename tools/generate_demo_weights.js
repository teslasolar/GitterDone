#!/usr/bin/env node
/**
 * Generate demo weights for FemtoLLM model
 * Creates random initialized weights for testing the WebGPU inference
 *
 * Usage: node generate_demo_weights.js
 */

const fs = require('fs')
const path = require('path')

// Configuration
const CONFIG = {
  vocabSize: 256,
  embedDim: 64,
  hiddenDim: 128,
  numHeads: 4,
  numLayers: 2,
  maxSeqLen: 128
}

// Seeded random for reproducibility
let seed = 42
function random() {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff
  return seed / 0x7fffffff
}

// Xavier/Glorot initialization
function xavier(fanIn, fanOut) {
  const stddev = Math.sqrt(2.0 / (fanIn + fanOut))
  return (random() - 0.5) * 2 * stddev
}

// Generate weight buffer
function generateWeights() {
  const weights = []

  console.log('Generating FemtoLLM weights...')

  // 1. Token embeddings: [vocab_size, embed_dim]
  console.log(`  Token embeddings: ${CONFIG.vocabSize} x ${CONFIG.embedDim}`)
  for (let i = 0; i < CONFIG.vocabSize * CONFIG.embedDim; i++) {
    weights.push(xavier(CONFIG.vocabSize, CONFIG.embedDim))
  }

  // 2. LayerNorm 1: gamma and beta [embed_dim]
  console.log(`  LayerNorm 1: ${CONFIG.embedDim}`)
  for (let i = 0; i < CONFIG.embedDim; i++) weights.push(1.0) // gamma
  for (let i = 0; i < CONFIG.embedDim; i++) weights.push(0.0) // beta

  // 3. Attention weights: Q, K, V projections [embed_dim, embed_dim * 3]
  console.log(`  Attention: ${CONFIG.embedDim} x ${CONFIG.embedDim * 3}`)
  for (let i = 0; i < CONFIG.embedDim * CONFIG.embedDim * 3; i++) {
    weights.push(xavier(CONFIG.embedDim, CONFIG.embedDim))
  }

  // Attention output projection [embed_dim, embed_dim]
  for (let i = 0; i < CONFIG.embedDim * CONFIG.embedDim; i++) {
    weights.push(xavier(CONFIG.embedDim, CONFIG.embedDim))
  }

  // 4. LayerNorm 2: gamma and beta [embed_dim]
  console.log(`  LayerNorm 2: ${CONFIG.embedDim}`)
  for (let i = 0; i < CONFIG.embedDim; i++) weights.push(1.0)
  for (let i = 0; i < CONFIG.embedDim; i++) weights.push(0.0)

  // 5. FFN: up projection [embed_dim, hidden_dim]
  console.log(`  FFN Up: ${CONFIG.embedDim} x ${CONFIG.hiddenDim}`)
  for (let i = 0; i < CONFIG.embedDim * CONFIG.hiddenDim; i++) {
    weights.push(xavier(CONFIG.embedDim, CONFIG.hiddenDim))
  }
  // bias
  for (let i = 0; i < CONFIG.hiddenDim; i++) weights.push(0.0)

  // 6. FFN: down projection [hidden_dim, embed_dim]
  console.log(`  FFN Down: ${CONFIG.hiddenDim} x ${CONFIG.embedDim}`)
  for (let i = 0; i < CONFIG.hiddenDim * CONFIG.embedDim; i++) {
    weights.push(xavier(CONFIG.hiddenDim, CONFIG.embedDim))
  }
  // bias
  for (let i = 0; i < CONFIG.embedDim; i++) weights.push(0.0)

  // 7. LM Head: [embed_dim, vocab_size]
  console.log(`  LM Head: ${CONFIG.embedDim} x ${CONFIG.vocabSize}`)
  for (let i = 0; i < CONFIG.embedDim * CONFIG.vocabSize; i++) {
    weights.push(xavier(CONFIG.embedDim, CONFIG.vocabSize))
  }

  return weights
}

// Convert to binary
function toFloat32Binary(weights) {
  const buffer = Buffer.alloc(weights.length * 4)
  for (let i = 0; i < weights.length; i++) {
    buffer.writeFloatLE(weights[i], i * 4)
  }
  return buffer
}

// Main
function main() {
  const outputDir = path.join(__dirname, '..', 'models', 'femto')

  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true })
  }

  // Generate weights
  const weights = generateWeights()
  console.log(`\nTotal weights: ${weights.length} floats (${(weights.length * 4 / 1024).toFixed(1)} KB)`)

  // Convert to binary
  const buffer = toFloat32Binary(weights)

  // Write shard
  const shardPath = path.join(outputDir, 'shard_000.bin')
  fs.writeFileSync(shardPath, buffer)
  console.log(`\nWritten: ${shardPath} (${buffer.length} bytes)`)

  // Update config with actual sizes
  const configPath = path.join(outputDir, 'config.json')
  if (fs.existsSync(configPath)) {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))
    config.shard_size = buffer.length
    config.parameters = weights.length
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2))
    console.log(`Updated: ${configPath}`)
  }

  console.log('\nDone! Demo weights generated successfully.')
}

main()
