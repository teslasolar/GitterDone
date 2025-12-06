/**
 * WebGPU Model Runner - Main Entry Point
 *
 * Run ML models directly in the browser with GPU acceleration.
 */

// Core modules
export { GPUCore } from './gpu-core.js'
export { TensorCube } from './tensor-cube.js'
export { ShardArray } from './shard-array.js'
export { ModelCore } from './model-core.js'
export { Pipeline } from './pipeline.js'

// Visualization & distribution
export { ModelSpace } from './model-space.js'
export { DistributedGPU } from './distributed-gpu.js'

// Model creation & optimization
export { ModelBuilder } from './model-builder.js'
export { Quantization } from './quantization.js'
export { ModelHub } from './model-hub.js'
export { FineTuner } from './fine-tuner.js'

// Quick setup function
export async function createModelRunner(modelUrl, options = {}) {
  const { GPUCore } = await import('./gpu-core.js')
  const { ModelCore } = await import('./model-core.js')
  const { Pipeline } = await import('./pipeline.js')

  const gpu = new GPUCore()
  await gpu.init()

  const model = new ModelCore(gpu)
  await model.init(modelUrl)

  const pipeline = new Pipeline(model)
  await pipeline.initTokenizer()

  return {
    gpu,
    model,
    pipeline,

    async generate(prompt, opts = {}) {
      const tokens = []
      for await (const token of pipeline.generate(prompt, { ...options, ...opts })) {
        tokens.push(token)
      }
      return tokens.join('')
    },

    async *stream(prompt, opts = {}) {
      yield* pipeline.generate(prompt, { ...options, ...opts })
    },

    destroy() {
      pipeline.clearCache()
      model.destroy()
      gpu.destroy()
    }
  }
}

// Version
export const VERSION = '1.0.0'
