# WebGPU Model Runner

Run ML models directly in your browser with GPU acceleration via WebGPU.

```
 GPU STATUS               MODEL I/O                    PERFORMANCE
┌───────────────────┐   ┌──────────────────────────┐  ┌──────────────────┐
│ Adapter: RTX 4090 │   │ > Enter prompt here...   │  │ Tokens/s: 127.3  │
│ Max Buffer: 16GB  │   │                          │  │ GPU Time: 42ms   │
│ Workgroups: 65535 │   │ [Generate] [Stop]        │  │ Memory: 256MB    │
├───────────────────┤   │                          │  ├──────────────────┤
│    Tensor Viz     │   │ Output:                  │  │ Temperature: 0.8 │
│  ████████████████ │   │ The quick brown fox...   │  │ Max Tokens: 100  │
│  ████████████████ │   │                          │  │ Top-K: 40        │
└───────────────────┘   └──────────────────────────┘  └──────────────────┘
```

## Features

- **WebGPU Compute**: Direct GPU access for 100+ GFLOPS inference
- **Model Sharding**: Git LFS support for models up to 7GB+
- **Streaming Inference**: Real-time token generation
- **3D Weight Navigation**: Visual exploration of model layers
- **Zero Backend**: Everything runs client-side

## Quick Start

### 1. Clone and serve locally

```bash
git clone https://github.com/yourusername/webgpu-model-runner
cd webgpu-model-runner
npx serve .
# Open http://localhost:3000
```

### 2. Or deploy to GitHub Pages

```bash
# Enable Git LFS
git lfs install
git lfs track "*.bin"

# Push to GitHub
git add -A && git commit -m "Initial commit"
git push origin main

# Settings → Pages → Deploy from branch → main
```

### 3. Access your model runner

```
https://[username].github.io/webgpu-model-runner
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     index.html                          │
├──────────────┬──────────────┬──────────────┬───────────┤
│   GPUCore    │  TensorCube  │  ShardArray  │  Pipeline │
│   (WebGPU)   │  (3D Ops)    │  (Git LFS)   │ (Stream)  │
├──────────────┴──────────────┴──────────────┴───────────┤
│                     ModelCore                           │
│              (ONNX/Custom Model Runner)                 │
├─────────────────────────────────────────────────────────┤
│                    WGSL Kernels                         │
│     matmul.wgsl │ attention.wgsl │ layernorm.wgsl      │
└─────────────────────────────────────────────────────────┘
```

## Supported Models

| Model | Size | Shards | Use Case |
|-------|------|--------|----------|
| FemtoLLM | 50KB | 1 | Demo/Testing |
| TinyLLM | 5MB | 1 | Fast inference |
| SmallLLM | 50MB | 1 | Balanced |
| BERT-base | 400MB | 8 | NLU tasks |
| Whisper | 1.5GB | 30 | Speech-to-text |
| Llama-7B | 7GB | 140 | Text generation |

## Adding Your Own Model

### 1. Convert to ONNX (optional)

```bash
# TensorFlow/Keras
python -m tf2onnx.convert --saved-model ./model --output model.onnx

# PyTorch
python export_onnx.py --model model.pt --output model.onnx
```

### 2. Shard the weights

```bash
python tools/shard_model.py model.onnx --output ./models/mymodel --size 50MB
```

### 3. Create config.json

```json
{
  "name": "MyModel",
  "version": "1.0",
  "parameters": 1000000,
  "vocab_size": 50000,
  "embed_dim": 768,
  "layers": [
    {"name": "embed", "type": "embedding", ...},
    {"name": "layer0", "type": "attention", ...}
  ]
}
```

### 4. Push with Git LFS

```bash
git lfs track "*.bin"
git add models/mymodel
git commit -m "Add MyModel"
git push
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| First token | <100ms | After model load |
| Tokens/sec | 50-200 | Depends on GPU |
| Memory | <4GB | Browser limit |
| Shard load | <1s/50MB | Cached after first load |

## Browser Requirements

- Chrome 113+ or Edge 113+
- WebGPU enabled (check `chrome://flags/#enable-webgpu`)
- 4GB+ VRAM recommended

## File Structure

```
/
├── index.html           # Main UI
├── src/
│   ├── gpu-core.js      # WebGPU initialization
│   ├── tensor-cube.js   # Tensor operations
│   ├── shard-array.js   # Model weight loading
│   ├── model-core.js    # Model execution
│   ├── pipeline.js      # Text generation
│   ├── model-space.js   # 3D visualization
│   └── distributed-gpu.js # Multi-tab inference
├── kernels/
│   ├── matmul.wgsl      # Matrix multiplication
│   ├── attention.wgsl   # Self-attention
│   ├── ffn.wgsl         # Feed-forward
│   ├── layernorm.wgsl   # Layer normalization
│   ├── embedding.wgsl   # Token embeddings
│   └── softmax.wgsl     # Softmax
├── models/
│   └── femto/
│       ├── config.json
│       ├── tokenizer.json
│       └── shard_000.bin
└── tools/
    ├── shard_model.py   # Model sharding
    └── generate_demo_weights.js
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+Enter | Generate |
| Escape | Stop generation |
| W/A/S/D | Navigate tensor |
| Q/E | Change layer |
| R/F | Change attention head |
| +/- | Zoom |

## API Usage

```javascript
import { GPUCore } from './src/gpu-core.js'
import { ModelCore } from './src/model-core.js'
import { Pipeline } from './src/pipeline.js'

// Initialize
const gpu = new GPUCore()
await gpu.init()

const model = new ModelCore(gpu)
await model.init('./models/femto')

const pipeline = new Pipeline(model)
await pipeline.initTokenizer()

// Generate
for await (const token of pipeline.generate('Hello', { maxTokens: 50 })) {
  console.log(token)
}
```

## Contributing

PRs welcome! Areas of interest:

- Additional WGSL kernels (flash attention, grouped query attention)
- More model format support (SafeTensors, GGUF)
- Quantization (INT8, INT4)
- Model caching improvements

## License

MIT
