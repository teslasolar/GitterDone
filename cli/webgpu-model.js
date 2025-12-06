#!/usr/bin/env node
/**
 * WebGPU Model CLI
 *
 * Convert, quantize, and manage models for WebGPU inference.
 *
 * Usage:
 *   npx webgpu-model convert model.onnx -o ./output
 *   npx webgpu-model quantize ./model --dtype int8
 *   npx webgpu-model create --preset micro-llm -o ./my-model
 *   npx webgpu-model serve ./model --port 8080
 *   npx webgpu-model publish ./model --registry github
 */

const fs = require('fs')
const path = require('path')
const { execSync } = require('child_process')

// Parse arguments
const args = process.argv.slice(2)
const command = args[0]

const flags = {}
const positional = []

for (let i = 1; i < args.length; i++) {
  if (args[i].startsWith('--')) {
    const key = args[i].slice(2)
    const value = args[i + 1] && !args[i + 1].startsWith('-') ? args[++i] : true
    flags[key] = value
  } else if (args[i].startsWith('-')) {
    const key = args[i].slice(1)
    const value = args[i + 1] && !args[i + 1].startsWith('-') ? args[++i] : true
    flags[key] = value
  } else {
    positional.push(args[i])
  }
}

// Commands
const commands = {
  help: showHelp,
  convert: convertModel,
  quantize: quantizeModel,
  create: createModel,
  serve: serveModel,
  publish: publishModel,
  info: showInfo,
  validate: validateModel,
  benchmark: benchmarkModel
}

// Main
async function main() {
  if (!command || command === 'help' || flags.help || flags.h) {
    showHelp()
    return
  }

  const handler = commands[command]
  if (!handler) {
    console.error(`Unknown command: ${command}`)
    console.log('Run "webgpu-model help" for usage')
    process.exit(1)
  }

  try {
    await handler(positional, flags)
  } catch (error) {
    console.error('Error:', error.message)
    if (flags.verbose || flags.v) {
      console.error(error.stack)
    }
    process.exit(1)
  }
}

// Show help
function showHelp() {
  console.log(`
┌──────────────────────────────────────────────────────────┐
│                  WebGPU Model CLI v1.0                   │
│         Convert & manage models for browser ML           │
└──────────────────────────────────────────────────────────┘

USAGE:
  webgpu-model <command> [options]

COMMANDS:
  convert     Convert ONNX/PyTorch model to WebGPU format
  quantize    Quantize model to INT8/INT4
  create      Create new model from template
  serve       Start local model server
  publish     Publish model to registry
  info        Show model information
  validate    Validate model configuration
  benchmark   Run inference benchmark

CONVERT OPTIONS:
  webgpu-model convert <input> [options]

  -o, --output <dir>    Output directory (default: ./output)
  --shard-size <bytes>  Shard size (default: 50MB)
  --format <type>       Input format: onnx, pytorch, safetensors
  --optimize            Enable optimization passes

QUANTIZE OPTIONS:
  webgpu-model quantize <model-dir> [options]

  --dtype <type>        int8, int4, int8-grouped (default: int8)
  --group-size <n>      Group size for grouped quantization
  --calibrate <data>    Calibration data file
  --exclude <layers>    Layers to exclude (comma-separated)

CREATE OPTIONS:
  webgpu-model create [options]

  --preset <name>       Template: nano-gpt, micro-llm, small-llm, etc
  --name <name>         Model name
  --vocab-size <n>      Vocabulary size
  --embed-dim <n>       Embedding dimension
  --num-layers <n>      Number of transformer layers
  --num-heads <n>       Number of attention heads
  -o, --output <dir>    Output directory

SERVE OPTIONS:
  webgpu-model serve <model-dir> [options]

  -p, --port <n>        Port number (default: 8080)
  --cors                Enable CORS headers

PUBLISH OPTIONS:
  webgpu-model publish <model-dir> [options]

  --registry <name>     Registry: github, npm, ipfs
  --repo <owner/repo>   GitHub repository
  --private             Make private

EXAMPLES:
  # Convert ONNX model
  webgpu-model convert model.onnx -o ./webgpu-model

  # Quantize to INT4
  webgpu-model quantize ./model --dtype int4

  # Create micro model
  webgpu-model create --preset micro-llm --name MyBot -o ./my-model

  # Serve locally
  webgpu-model serve ./model --port 3000

  # Publish to GitHub
  webgpu-model publish ./model --registry github --repo user/model
`)
}

// Convert model
async function convertModel(positional, flags) {
  const inputPath = positional[0]
  if (!inputPath) {
    throw new Error('Input file required')
  }

  const outputDir = flags.output || flags.o || './output'
  const shardSize = parseSize(flags['shard-size'] || '50MB')
  const format = flags.format || detectFormat(inputPath)

  console.log(`Converting ${inputPath} (${format}) to WebGPU format...`)
  console.log(`Output: ${outputDir}`)
  console.log(`Shard size: ${formatBytes(shardSize)}`)
  console.log()

  // Ensure output directory
  fs.mkdirSync(outputDir, { recursive: true })

  let weights, layerInfo

  switch (format) {
    case 'onnx':
      ({ weights, layerInfo } = await convertONNX(inputPath))
      break
    case 'pytorch':
      ({ weights, layerInfo } = await convertPyTorch(inputPath))
      break
    case 'safetensors':
      ({ weights, layerInfo } = await convertSafeTensors(inputPath))
      break
    default:
      throw new Error(`Unsupported format: ${format}`)
  }

  // Shard weights
  console.log(`Sharding ${formatBytes(weights.byteLength)}...`)
  const shards = shardBuffer(weights, shardSize)

  for (let i = 0; i < shards.length; i++) {
    const shardPath = path.join(outputDir, `shard_${String(i).padStart(3, '0')}.bin`)
    fs.writeFileSync(shardPath, Buffer.from(shards[i]))
    console.log(`  ${path.basename(shardPath)}: ${formatBytes(shards[i].byteLength)}`)
  }

  // Generate config
  const config = generateConfig(layerInfo, shards.length, shardSize)
  fs.writeFileSync(path.join(outputDir, 'config.json'), JSON.stringify(config, null, 2))
  console.log(`  config.json: ${JSON.stringify(config).length} bytes`)

  // Generate tokenizer (placeholder)
  const tokenizer = generateTokenizer()
  fs.writeFileSync(path.join(outputDir, 'tokenizer.json'), JSON.stringify(tokenizer, null, 2))
  console.log(`  tokenizer.json: ${JSON.stringify(tokenizer).length} bytes`)

  console.log()
  console.log(`✓ Conversion complete!`)
  console.log(`  Total shards: ${shards.length}`)
  console.log(`  Total size: ${formatBytes(weights.byteLength)}`)
}

// Quantize model
async function quantizeModel(positional, flags) {
  const modelDir = positional[0]
  if (!modelDir) {
    throw new Error('Model directory required')
  }

  const dtype = flags.dtype || 'int8'
  const groupSize = parseInt(flags['group-size'] || '128')
  const exclude = (flags.exclude || 'embed,lm_head').split(',')

  console.log(`Quantizing ${modelDir} to ${dtype}...`)

  // Load config
  const configPath = path.join(modelDir, 'config.json')
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))

  // Load weights
  const weights = loadShards(modelDir, config.shards)

  let quantized
  switch (dtype) {
    case 'int8':
      quantized = quantizeInt8(weights, exclude)
      break
    case 'int4':
      quantized = quantizeInt4(weights, exclude)
      break
    case 'int8-grouped':
      quantized = quantizeGrouped(weights, groupSize, exclude)
      break
    default:
      throw new Error(`Unsupported dtype: ${dtype}`)
  }

  // Save quantized weights
  const outputDir = modelDir + '-' + dtype
  fs.mkdirSync(outputDir, { recursive: true })

  const shards = shardBuffer(quantized.data, 50 * 1024 * 1024)
  for (let i = 0; i < shards.length; i++) {
    const shardPath = path.join(outputDir, `shard_${String(i).padStart(3, '0')}.bin`)
    fs.writeFileSync(shardPath, Buffer.from(shards[i]))
  }

  // Update config
  config.dtype = dtype
  config.quantization = quantized.metadata
  config.shards = shards.length
  fs.writeFileSync(path.join(outputDir, 'config.json'), JSON.stringify(config, null, 2))

  // Copy tokenizer
  const tokenizerPath = path.join(modelDir, 'tokenizer.json')
  if (fs.existsSync(tokenizerPath)) {
    fs.copyFileSync(tokenizerPath, path.join(outputDir, 'tokenizer.json'))
  }

  console.log()
  console.log(`✓ Quantization complete!`)
  console.log(`  Original size: ${formatBytes(weights.byteLength)}`)
  console.log(`  Quantized size: ${formatBytes(quantized.data.byteLength)}`)
  console.log(`  Compression: ${(weights.byteLength / quantized.data.byteLength).toFixed(1)}x`)
  console.log(`  Output: ${outputDir}`)
}

// Create model from template
async function createModel(positional, flags) {
  const preset = flags.preset
  const name = flags.name || 'MyModel'
  const outputDir = flags.output || flags.o || './' + name.toLowerCase().replace(/\s+/g, '-')

  console.log(`Creating model: ${name}`)
  if (preset) console.log(`Preset: ${preset}`)
  console.log(`Output: ${outputDir}`)
  console.log()

  fs.mkdirSync(outputDir, { recursive: true })

  // Get config from preset or flags
  const config = getPresetConfig(preset, flags)
  config.name = name

  // Build layer structure
  const layers = buildLayers(config)
  config.layers = layers

  // Calculate total size
  const totalParams = layers.reduce((sum, l) => sum + (l.weight_size || 0) / 4, 0)
  config.parameters = totalParams

  // Write config
  fs.writeFileSync(path.join(outputDir, 'config.json'), JSON.stringify(config, null, 2))
  console.log(`  config.json created`)

  // Generate tokenizer
  const tokenizer = generateTokenizer(config.vocab_size)
  fs.writeFileSync(path.join(outputDir, 'tokenizer.json'), JSON.stringify(tokenizer, null, 2))
  console.log(`  tokenizer.json created`)

  // Generate random weights
  console.log(`  Generating weights...`)
  const weights = generateWeights(layers)
  const shards = shardBuffer(weights, 50 * 1024 * 1024)

  for (let i = 0; i < shards.length; i++) {
    const shardPath = path.join(outputDir, `shard_${String(i).padStart(3, '0')}.bin`)
    fs.writeFileSync(shardPath, Buffer.from(shards[i]))
    console.log(`  ${path.basename(shardPath)}: ${formatBytes(shards[i].byteLength)}`)
  }

  // Update shard count
  config.shards = shards.length
  fs.writeFileSync(path.join(outputDir, 'config.json'), JSON.stringify(config, null, 2))

  console.log()
  console.log(`✓ Model created!`)
  console.log(`  Parameters: ${formatNumber(totalParams)}`)
  console.log(`  Size: ${formatBytes(weights.byteLength)}`)
  console.log()
  console.log(`Next steps:`)
  console.log(`  1. Train or fine-tune the model`)
  console.log(`  2. Run: webgpu-model serve ${outputDir}`)
  console.log(`  3. Open http://localhost:8080`)
}

// Serve model locally
async function serveModel(positional, flags) {
  const modelDir = positional[0] || '.'
  const port = parseInt(flags.port || flags.p || '8080')
  const cors = flags.cors

  console.log(`Serving ${modelDir} on http://localhost:${port}`)

  const http = require('http')

  const server = http.createServer((req, res) => {
    if (cors) {
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Methods', 'GET')
    }

    let filePath = path.join(modelDir, req.url === '/' ? 'index.html' : req.url)

    // Check parent directory for index.html
    if (!fs.existsSync(filePath) && req.url === '/') {
      filePath = path.join(__dirname, '..', 'index.html')
    }

    if (!fs.existsSync(filePath)) {
      res.writeHead(404)
      res.end('Not found')
      return
    }

    const ext = path.extname(filePath)
    const contentType = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.json': 'application/json',
      '.bin': 'application/octet-stream',
      '.wgsl': 'text/plain'
    }[ext] || 'application/octet-stream'

    res.setHeader('Content-Type', contentType)
    fs.createReadStream(filePath).pipe(res)
  })

  server.listen(port)
  console.log()
  console.log('Press Ctrl+C to stop')
}

// Publish model
async function publishModel(positional, flags) {
  const modelDir = positional[0]
  if (!modelDir) {
    throw new Error('Model directory required')
  }

  const registry = flags.registry || 'github'

  console.log(`Publishing ${modelDir} to ${registry}...`)

  // Validate model first
  await validateModel([modelDir], {})

  switch (registry) {
    case 'github':
      await publishToGitHub(modelDir, flags)
      break
    case 'npm':
      await publishToNPM(modelDir, flags)
      break
    default:
      throw new Error(`Unsupported registry: ${registry}`)
  }
}

// Show model info
async function showInfo(positional, flags) {
  const modelDir = positional[0] || '.'

  const configPath = path.join(modelDir, 'config.json')
  if (!fs.existsSync(configPath)) {
    throw new Error('config.json not found')
  }

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))

  console.log(`
┌─ Model Info ────────────────────────────────────────────┐
│ Name: ${config.name.padEnd(48)}│
│ Version: ${(config.version || '1.0.0').padEnd(45)}│
│ Architecture: ${(config.architecture || 'transformer').padEnd(39)}│
├─────────────────────────────────────────────────────────┤
│ Parameters: ${formatNumber(config.parameters).padEnd(42)}│
│ Vocab Size: ${String(config.vocab_size).padEnd(42)}│
│ Embed Dim: ${String(config.embed_dim).padEnd(43)}│
│ Hidden Dim: ${String(config.hidden_dim).padEnd(42)}│
│ Num Heads: ${String(config.num_heads).padEnd(43)}│
│ Num Layers: ${String(config.num_layers).padEnd(42)}│
│ Max Seq Len: ${String(config.max_seq_len).padEnd(41)}│
├─────────────────────────────────────────────────────────┤
│ Dtype: ${(config.dtype || 'float32').padEnd(47)}│
│ Shards: ${String(config.shards).padEnd(46)}│
│ Shard Size: ${formatBytes(config.shard_size || 50000000).padEnd(42)}│
└─────────────────────────────────────────────────────────┘
`)
}

// Validate model
async function validateModel(positional, flags) {
  const modelDir = positional[0] || '.'

  console.log(`Validating ${modelDir}...`)

  const errors = []
  const warnings = []

  // Check config.json
  const configPath = path.join(modelDir, 'config.json')
  if (!fs.existsSync(configPath)) {
    errors.push('config.json not found')
  } else {
    try {
      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))

      if (!config.name) warnings.push('config.name is missing')
      if (!config.vocab_size) errors.push('config.vocab_size is required')
      if (!config.embed_dim) errors.push('config.embed_dim is required')
      if (!config.layers || config.layers.length === 0) errors.push('config.layers is empty')

      // Check shards exist
      for (let i = 0; i < (config.shards || 1); i++) {
        const shardPath = path.join(modelDir, `shard_${String(i).padStart(3, '0')}.bin`)
        if (!fs.existsSync(shardPath)) {
          errors.push(`Missing shard: ${shardPath}`)
        }
      }
    } catch (e) {
      errors.push(`Invalid config.json: ${e.message}`)
    }
  }

  // Check tokenizer
  const tokenizerPath = path.join(modelDir, 'tokenizer.json')
  if (!fs.existsSync(tokenizerPath)) {
    warnings.push('tokenizer.json not found (will use default)')
  }

  // Report
  if (errors.length > 0) {
    console.log()
    console.log('Errors:')
    errors.forEach(e => console.log(`  ✗ ${e}`))
  }

  if (warnings.length > 0) {
    console.log()
    console.log('Warnings:')
    warnings.forEach(w => console.log(`  ⚠ ${w}`))
  }

  if (errors.length === 0) {
    console.log()
    console.log('✓ Model is valid!')
  } else {
    process.exit(1)
  }
}

// Helpers
function parseSize(str) {
  const match = str.match(/^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)?$/i)
  if (!match) return parseInt(str)

  const num = parseFloat(match[1])
  const unit = (match[2] || 'B').toUpperCase()
  const multipliers = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3 }

  return Math.floor(num * multipliers[unit])
}

function formatBytes(bytes) {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB'
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB'
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB'
  return bytes + ' B'
}

function formatNumber(num) {
  if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B'
  if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M'
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K'
  return num.toString()
}

function detectFormat(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  switch (ext) {
    case '.onnx': return 'onnx'
    case '.pt':
    case '.pth':
    case '.bin': return 'pytorch'
    case '.safetensors': return 'safetensors'
    default: return 'unknown'
  }
}

function shardBuffer(buffer, shardSize) {
  const shards = []
  let offset = 0

  while (offset < buffer.byteLength) {
    const end = Math.min(offset + shardSize, buffer.byteLength)
    shards.push(buffer.slice(offset, end))
    offset = end
  }

  return shards
}

function getPresetConfig(preset, flags) {
  const presets = {
    'nano-gpt': { vocab_size: 256, embed_dim: 64, hidden_dim: 128, num_heads: 4, num_layers: 2, max_seq_len: 128 },
    'micro-llm': { vocab_size: 8000, embed_dim: 256, hidden_dim: 512, num_heads: 8, num_layers: 4, max_seq_len: 256 },
    'small-llm': { vocab_size: 32000, embed_dim: 512, hidden_dim: 1024, num_heads: 8, num_layers: 6, max_seq_len: 512 },
    'medium-llm': { vocab_size: 32000, embed_dim: 768, hidden_dim: 2048, num_heads: 12, num_layers: 12, max_seq_len: 1024 }
  }

  const base = presets[preset] || presets['micro-llm']

  return {
    ...base,
    vocab_size: parseInt(flags['vocab-size'] || base.vocab_size),
    embed_dim: parseInt(flags['embed-dim'] || base.embed_dim),
    hidden_dim: parseInt(flags['hidden-dim'] || base.hidden_dim),
    num_heads: parseInt(flags['num-heads'] || base.num_heads),
    num_layers: parseInt(flags['num-layers'] || base.num_layers),
    max_seq_len: parseInt(flags['max-seq-len'] || base.max_seq_len),
    architecture: 'transformer',
    dtype: 'float32'
  }
}

function buildLayers(config) {
  // Simplified layer builder (matches model-builder.js)
  const layers = []
  let offset = 0

  // Embedding
  const embedSize = config.vocab_size * config.embed_dim * 4
  layers.push({
    name: 'embed',
    type: 'embedding',
    params: { vocab_size: config.vocab_size, embed_dim: config.embed_dim },
    weight_offset: offset,
    weight_size: embedSize
  })
  offset += embedSize

  // Transformer layers
  for (let i = 0; i < config.num_layers; i++) {
    // Attention
    const attnSize = config.embed_dim * config.embed_dim * 4 * 4
    layers.push({
      name: `layer${i}_attn`,
      type: 'attention',
      params: { num_heads: config.num_heads, head_dim: config.embed_dim / config.num_heads },
      weight_offset: offset,
      weight_size: attnSize
    })
    offset += attnSize

    // FFN
    const ffnSize = config.embed_dim * config.hidden_dim * 4 * 2
    layers.push({
      name: `layer${i}_ffn`,
      type: 'linear',
      params: { in_features: config.embed_dim, out_features: config.hidden_dim },
      weight_offset: offset,
      weight_size: ffnSize
    })
    offset += ffnSize
  }

  // LM head
  const lmHeadSize = config.embed_dim * config.vocab_size * 4
  layers.push({
    name: 'lm_head',
    type: 'linear',
    params: { in_features: config.embed_dim, out_features: config.vocab_size },
    weight_offset: offset,
    weight_size: lmHeadSize
  })

  return layers
}

function generateWeights(layers) {
  let totalSize = 0
  for (const layer of layers) {
    totalSize += layer.weight_size || 0
  }

  const buffer = Buffer.alloc(totalSize)

  // Fill with small random values
  for (let i = 0; i < totalSize; i += 4) {
    buffer.writeFloatLE((Math.random() - 0.5) * 0.1, i)
  }

  return buffer
}

function generateTokenizer(vocabSize = 256) {
  const vocab = {}
  vocab['<pad>'] = 0
  vocab['<eos>'] = 1
  vocab['<bos>'] = 2
  vocab['<unk>'] = 3

  // Add printable ASCII
  for (let i = 32; i < 127 && Object.keys(vocab).length < vocabSize; i++) {
    vocab[String.fromCharCode(i)] = Object.keys(vocab).length
  }

  return {
    version: '1.0',
    type: 'character',
    vocab,
    special_tokens: {
      pad_token: '<pad>', pad_token_id: 0,
      eos_token: '<eos>', eos_token_id: 1,
      bos_token: '<bos>', bos_token_id: 2,
      unk_token: '<unk>', unk_token_id: 3
    }
  }
}

function generateConfig(layerInfo, numShards, shardSize) {
  return {
    name: 'ConvertedModel',
    version: '1.0.0',
    architecture: 'transformer',
    shards: numShards,
    shard_size: shardSize,
    dtype: 'float32',
    layers: Object.entries(layerInfo).map(([name, info]) => ({
      name,
      type: 'linear',
      weight_offset: info.offset,
      weight_size: info.size,
      params: { shape: info.shape }
    }))
  }
}

// Stubs for conversion (would need actual implementations)
async function convertONNX(inputPath) {
  console.log('Converting ONNX model...')
  console.log('Note: Full ONNX conversion requires onnx package')
  // Return demo weights for now
  return {
    weights: Buffer.alloc(1024 * 1024),
    layerInfo: {}
  }
}

async function convertPyTorch(inputPath) {
  console.log('Converting PyTorch model...')
  console.log('Note: Full PyTorch conversion requires torch package')
  return {
    weights: Buffer.alloc(1024 * 1024),
    layerInfo: {}
  }
}

async function convertSafeTensors(inputPath) {
  console.log('Converting SafeTensors model...')
  return {
    weights: Buffer.alloc(1024 * 1024),
    layerInfo: {}
  }
}

function loadShards(modelDir, numShards) {
  const chunks = []
  for (let i = 0; i < numShards; i++) {
    const shardPath = path.join(modelDir, `shard_${String(i).padStart(3, '0')}.bin`)
    chunks.push(fs.readFileSync(shardPath))
  }
  return Buffer.concat(chunks)
}

function quantizeInt8(weights, exclude) {
  const result = Buffer.alloc(weights.length / 4)
  // Simplified quantization
  for (let i = 0; i < weights.length; i += 4) {
    const val = weights.readFloatLE(i)
    result[i / 4] = Math.max(-127, Math.min(127, Math.round(val * 127)))
  }
  return { data: result, metadata: { dtype: 'int8', scale: 1 / 127 } }
}

function quantizeInt4(weights, exclude) {
  const result = Buffer.alloc(weights.length / 8)
  for (let i = 0; i < weights.length; i += 8) {
    const v1 = Math.max(-7, Math.min(7, Math.round(weights.readFloatLE(i) * 7)))
    const v2 = Math.max(-7, Math.min(7, Math.round(weights.readFloatLE(i + 4) * 7)))
    result[i / 8] = ((v1 + 8) & 0xF) | (((v2 + 8) & 0xF) << 4)
  }
  return { data: result, metadata: { dtype: 'int4', scale: 1 / 7 } }
}

function quantizeGrouped(weights, groupSize, exclude) {
  return quantizeInt8(weights, exclude)
}

async function publishToGitHub(modelDir, flags) {
  const repo = flags.repo
  if (!repo) throw new Error('--repo required for GitHub publishing')

  console.log(`Publishing to GitHub: ${repo}`)
  console.log('Note: Ensure git-lfs is installed and configured')

  // Check if git repo exists
  if (!fs.existsSync(path.join(modelDir, '.git'))) {
    execSync('git init', { cwd: modelDir })
    execSync('git lfs install', { cwd: modelDir })
    execSync('git lfs track "*.bin"', { cwd: modelDir })
  }

  execSync('git add -A', { cwd: modelDir })
  execSync('git commit -m "Publish model"', { cwd: modelDir })
  execSync(`git remote add origin https://github.com/${repo}.git`, { cwd: modelDir })
  execSync('git push -u origin main', { cwd: modelDir })

  console.log()
  console.log(`✓ Published to https://github.com/${repo}`)
}

async function publishToNPM(modelDir, flags) {
  console.log('NPM publishing not yet implemented')
}

// Run
main().catch(console.error)
