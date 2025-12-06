/**
 * Model Templates Library
 * Pre-configured architectures for different use cases
 */

export const ModelTemplates = {
  // ============================================
  // TEXT GENERATION MODELS
  // ============================================
  text: {
    'nano-gpt': {
      name: 'NanoGPT',
      category: 'text',
      description: 'Tiny GPT for learning and edge devices',
      size: '500KB',
      inference: '<10ms',
      config: {
        architecture: 'gpt',
        vocab_size: 256,
        embed_dim: 64,
        hidden_dim: 128,
        num_heads: 4,
        num_layers: 2,
        max_seq_len: 128,
        activation: 'gelu',
        norm_type: 'layernorm',
        tie_embeddings: true
      },
      useCases: ['Learning', 'Demos', 'Edge devices', 'IoT']
    },

    'micro-gpt': {
      name: 'MicroGPT',
      category: 'text',
      description: 'Small but capable text generation',
      size: '5MB',
      inference: '20-50ms',
      config: {
        architecture: 'gpt',
        vocab_size: 8000,
        embed_dim: 256,
        hidden_dim: 512,
        num_heads: 8,
        num_layers: 4,
        max_seq_len: 256,
        activation: 'gelu',
        norm_type: 'layernorm',
        dropout: 0.1
      },
      useCases: ['Mobile apps', 'Browser', 'Simple chatbots']
    },

    'small-llm': {
      name: 'SmallLLM',
      category: 'text',
      description: 'Balanced performance for general tasks',
      size: '50MB',
      inference: '50-100ms',
      config: {
        architecture: 'llama',
        vocab_size: 32000,
        embed_dim: 512,
        hidden_dim: 1408,
        num_heads: 8,
        num_layers: 8,
        max_seq_len: 512,
        activation: 'silu',
        norm_type: 'rmsnorm',
        rope: true,
        rope_theta: 10000
      },
      useCases: ['Chatbots', 'Content generation', 'Summarization']
    },

    'medium-llm': {
      name: 'MediumLLM',
      category: 'text',
      description: 'Strong performance for production',
      size: '300MB',
      inference: '100-200ms',
      config: {
        architecture: 'llama',
        vocab_size: 32000,
        embed_dim: 768,
        hidden_dim: 2048,
        num_heads: 12,
        num_layers: 12,
        max_seq_len: 2048,
        activation: 'silu',
        norm_type: 'rmsnorm',
        rope: true,
        gqa_groups: 4
      },
      useCases: ['Production apps', 'Complex reasoning', 'Long-form content']
    },

    'large-llm': {
      name: 'LargeLLM',
      category: 'text',
      description: 'High-quality output (requires good GPU)',
      size: '1.5GB',
      inference: '200-500ms',
      config: {
        architecture: 'llama',
        vocab_size: 32000,
        embed_dim: 1024,
        hidden_dim: 4096,
        num_heads: 16,
        num_layers: 24,
        max_seq_len: 4096,
        activation: 'silu',
        norm_type: 'rmsnorm',
        rope: true,
        gqa_groups: 4,
        sliding_window: 2048
      },
      useCases: ['Enterprise', 'Research', 'Complex tasks']
    }
  },

  // ============================================
  // CODE MODELS
  // ============================================
  code: {
    'code-micro': {
      name: 'CodeMicro',
      category: 'code',
      description: 'Lightweight code completion',
      size: '10MB',
      inference: '30-60ms',
      config: {
        architecture: 'gpt',
        vocab_size: 50000,
        embed_dim: 256,
        hidden_dim: 512,
        num_heads: 8,
        num_layers: 4,
        max_seq_len: 512,
        activation: 'gelu',
        special_tokens: ['<|code|>', '<|/code|>', '<|comment|>']
      },
      useCases: ['Autocomplete', 'Simple suggestions']
    },

    'code-small': {
      name: 'CodeSmall',
      category: 'code',
      description: 'Capable code generation',
      size: '100MB',
      inference: '80-150ms',
      config: {
        architecture: 'llama',
        vocab_size: 50000,
        embed_dim: 512,
        hidden_dim: 2048,
        num_heads: 8,
        num_layers: 12,
        max_seq_len: 2048,
        activation: 'silu',
        norm_type: 'rmsnorm',
        rope: true,
        special_tokens: ['<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>']
      },
      useCases: ['Code completion', 'Function generation', 'Docstrings']
    },

    'code-medium': {
      name: 'CodeMedium',
      category: 'code',
      description: 'Full-featured code assistant',
      size: '500MB',
      inference: '150-300ms',
      config: {
        architecture: 'llama',
        vocab_size: 64000,
        embed_dim: 768,
        hidden_dim: 3072,
        num_heads: 12,
        num_layers: 16,
        max_seq_len: 8192,
        activation: 'silu',
        norm_type: 'rmsnorm',
        rope: true,
        rope_scaling: { type: 'dynamic', factor: 2.0 }
      },
      useCases: ['Full file generation', 'Refactoring', 'Bug fixing']
    }
  },

  // ============================================
  // CHAT/INSTRUCTION MODELS
  // ============================================
  chat: {
    'chat-nano': {
      name: 'ChatNano',
      category: 'chat',
      description: 'Ultra-light conversational model',
      size: '2MB',
      inference: '15-30ms',
      config: {
        architecture: 'gpt',
        vocab_size: 8000,
        embed_dim: 128,
        hidden_dim: 256,
        num_heads: 4,
        num_layers: 4,
        max_seq_len: 256,
        chat_template: 'chatml'
      },
      useCases: ['Simple Q&A', 'FAQ bots', 'Widgets']
    },

    'chat-small': {
      name: 'ChatSmall',
      category: 'chat',
      description: 'Responsive chat assistant',
      size: '50MB',
      inference: '40-80ms',
      config: {
        architecture: 'llama',
        vocab_size: 32000,
        embed_dim: 512,
        hidden_dim: 1536,
        num_heads: 8,
        num_layers: 8,
        max_seq_len: 2048,
        activation: 'silu',
        norm_type: 'rmsnorm',
        chat_template: 'llama2'
      },
      useCases: ['Customer support', 'Interactive assistants']
    },

    'chat-medium': {
      name: 'ChatMedium',
      category: 'chat',
      description: 'Advanced conversational AI',
      size: '200MB',
      inference: '80-150ms',
      config: {
        architecture: 'llama',
        vocab_size: 32000,
        embed_dim: 768,
        hidden_dim: 2048,
        num_heads: 12,
        num_layers: 12,
        max_seq_len: 4096,
        activation: 'silu',
        norm_type: 'rmsnorm',
        rope: true,
        chat_template: 'chatml'
      },
      useCases: ['Complex conversations', 'Multi-turn dialogue']
    }
  },

  // ============================================
  // EMBEDDING MODELS
  // ============================================
  embedding: {
    'embed-tiny': {
      name: 'EmbedTiny',
      category: 'embedding',
      description: 'Fast text embeddings',
      size: '5MB',
      inference: '5-10ms',
      config: {
        architecture: 'bert',
        vocab_size: 30000,
        embed_dim: 128,
        hidden_dim: 256,
        num_heads: 4,
        num_layers: 4,
        max_seq_len: 256,
        pooling: 'mean'
      },
      useCases: ['Search', 'Similarity', 'Classification']
    },

    'embed-small': {
      name: 'EmbedSmall',
      category: 'embedding',
      description: 'Quality embeddings for RAG',
      size: '30MB',
      inference: '10-20ms',
      config: {
        architecture: 'bert',
        vocab_size: 30000,
        embed_dim: 384,
        hidden_dim: 768,
        num_heads: 6,
        num_layers: 6,
        max_seq_len: 512,
        pooling: 'cls'
      },
      useCases: ['RAG', 'Semantic search', 'Clustering']
    }
  },

  // ============================================
  // VISION-LANGUAGE MODELS
  // ============================================
  vision: {
    'vision-tiny': {
      name: 'VisionTiny',
      category: 'vision',
      description: 'Basic image understanding',
      size: '20MB',
      inference: '50-100ms',
      config: {
        architecture: 'clip',
        vision_encoder: {
          type: 'vit',
          patch_size: 16,
          image_size: 224,
          embed_dim: 256,
          num_heads: 4,
          num_layers: 6
        },
        text_encoder: {
          vocab_size: 32000,
          embed_dim: 256,
          num_heads: 4,
          num_layers: 4
        }
      },
      useCases: ['Image classification', 'Simple captioning']
    }
  },

  // ============================================
  // SPECIALIZED MODELS
  // ============================================
  specialized: {
    'sql-generator': {
      name: 'SQLGen',
      category: 'specialized',
      description: 'Natural language to SQL',
      size: '30MB',
      inference: '30-60ms',
      config: {
        architecture: 'gpt',
        vocab_size: 20000,
        embed_dim: 384,
        hidden_dim: 768,
        num_heads: 6,
        num_layers: 6,
        max_seq_len: 512,
        special_tokens: ['<|schema|>', '<|query|>', '<|result|>']
      },
      useCases: ['Database queries', 'Data analysis']
    },

    'summarizer': {
      name: 'Summarizer',
      category: 'specialized',
      description: 'Text summarization',
      size: '40MB',
      inference: '40-80ms',
      config: {
        architecture: 'bart',
        vocab_size: 32000,
        embed_dim: 512,
        hidden_dim: 1024,
        num_heads: 8,
        num_layers: 6,
        max_seq_len: 1024,
        decoder_layers: 6
      },
      useCases: ['Article summaries', 'Meeting notes', 'TL;DR']
    },

    'sentiment': {
      name: 'SentimentAnalyzer',
      category: 'specialized',
      description: 'Sentiment classification',
      size: '10MB',
      inference: '5-15ms',
      config: {
        architecture: 'bert',
        vocab_size: 30000,
        embed_dim: 256,
        hidden_dim: 512,
        num_heads: 4,
        num_layers: 4,
        max_seq_len: 256,
        num_classes: 3,
        task: 'classification'
      },
      useCases: ['Review analysis', 'Social media', 'Feedback']
    },

    'ner': {
      name: 'EntityExtractor',
      category: 'specialized',
      description: 'Named entity recognition',
      size: '15MB',
      inference: '10-20ms',
      config: {
        architecture: 'bert',
        vocab_size: 30000,
        embed_dim: 256,
        hidden_dim: 512,
        num_heads: 4,
        num_layers: 4,
        max_seq_len: 256,
        num_labels: 9,
        task: 'token_classification'
      },
      useCases: ['Information extraction', 'Document processing']
    }
  }
}

/**
 * Get all templates as flat array
 */
export function getAllTemplates() {
  const templates = []
  for (const [category, models] of Object.entries(ModelTemplates)) {
    for (const [id, template] of Object.entries(models)) {
      templates.push({ id, ...template })
    }
  }
  return templates
}

/**
 * Get templates by category
 */
export function getTemplatesByCategory(category) {
  return ModelTemplates[category] || {}
}

/**
 * Get template by ID
 */
export function getTemplate(id) {
  for (const category of Object.values(ModelTemplates)) {
    if (category[id]) return { id, ...category[id] }
  }
  return null
}

/**
 * Filter templates by criteria
 */
export function filterTemplates(criteria = {}) {
  const { maxSize, category, useCase, minLayers, maxLayers } = criteria
  let templates = getAllTemplates()

  if (category) {
    templates = templates.filter(t => t.category === category)
  }

  if (maxSize) {
    const maxBytes = parseSize(maxSize)
    templates = templates.filter(t => parseSize(t.size) <= maxBytes)
  }

  if (useCase) {
    templates = templates.filter(t =>
      t.useCases?.some(u => u.toLowerCase().includes(useCase.toLowerCase()))
    )
  }

  if (minLayers) {
    templates = templates.filter(t => t.config.num_layers >= minLayers)
  }

  if (maxLayers) {
    templates = templates.filter(t => t.config.num_layers <= maxLayers)
  }

  return templates
}

/**
 * Parse size string to bytes
 */
function parseSize(sizeStr) {
  const match = sizeStr.match(/^([\d.]+)\s*(KB|MB|GB)?$/i)
  if (!match) return 0

  const num = parseFloat(match[1])
  const unit = (match[2] || 'B').toUpperCase()
  const multipliers = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3 }

  return num * multipliers[unit]
}

export default ModelTemplates
