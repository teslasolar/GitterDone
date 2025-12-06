/**
 * Training Data Templates
 * Data formats, augmentation strategies, and dataset configurations
 */

export const TrainingTemplates = {
  // ============================================
  // DATA FORMATS
  // ============================================
  formats: {
    'jsonl': {
      name: 'JSON Lines',
      description: 'One JSON object per line - most common format',
      extension: '.jsonl',
      example: `{"text": "Hello, how are you?"}
{"text": "I am doing well, thank you!"}
{"text": "The quick brown fox jumps over the lazy dog."}`,
      schema: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Raw text for pretraining' }
        },
        required: ['text']
      }
    },

    'chat': {
      name: 'Chat Format',
      description: 'Multi-turn conversations with roles',
      extension: '.jsonl',
      example: `{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help?"}]}
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}`,
      schema: {
        type: 'object',
        properties: {
          messages: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                role: { enum: ['system', 'user', 'assistant'] },
                content: { type: 'string' }
              }
            }
          }
        },
        required: ['messages']
      }
    },

    'instruction': {
      name: 'Instruction Format',
      description: 'Input-output pairs for instruction tuning',
      extension: '.jsonl',
      example: `{"instruction": "Summarize this text", "input": "The quick brown fox...", "output": "A fox jumped over a dog."}
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}`,
      schema: {
        type: 'object',
        properties: {
          instruction: { type: 'string' },
          input: { type: 'string' },
          output: { type: 'string' }
        },
        required: ['instruction', 'output']
      }
    },

    'qa': {
      name: 'Question-Answer',
      description: 'Q&A pairs with optional context',
      extension: '.jsonl',
      example: `{"question": "What is the capital of France?", "answer": "Paris"}
{"context": "The Eiffel Tower is in Paris.", "question": "Where is the Eiffel Tower?", "answer": "Paris"}`,
      schema: {
        type: 'object',
        properties: {
          context: { type: 'string' },
          question: { type: 'string' },
          answer: { type: 'string' }
        },
        required: ['question', 'answer']
      }
    },

    'preference': {
      name: 'Preference/DPO',
      description: 'Chosen vs rejected responses for alignment',
      extension: '.jsonl',
      example: `{"prompt": "Tell me a joke", "chosen": "Why did the chicken...", "rejected": "I don't know any jokes."}`,
      schema: {
        type: 'object',
        properties: {
          prompt: { type: 'string' },
          chosen: { type: 'string' },
          rejected: { type: 'string' }
        },
        required: ['prompt', 'chosen', 'rejected']
      }
    },

    'code': {
      name: 'Code Format',
      description: 'Code with metadata for code models',
      extension: '.jsonl',
      example: `{"code": "def hello():\\n    print('hello')", "language": "python", "docstring": "Prints hello"}
{"code": "function add(a, b) { return a + b; }", "language": "javascript", "context": "math utils"}`,
      schema: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          language: { type: 'string' },
          docstring: { type: 'string' },
          context: { type: 'string' }
        },
        required: ['code']
      }
    },

    'classification': {
      name: 'Classification',
      description: 'Text with labels for classification tasks',
      extension: '.jsonl',
      example: `{"text": "I love this product!", "label": "positive"}
{"text": "Terrible experience", "label": "negative"}`,
      schema: {
        type: 'object',
        properties: {
          text: { type: 'string' },
          label: { type: 'string' }
        },
        required: ['text', 'label']
      }
    }
  },

  // ============================================
  // DATASET TEMPLATES
  // ============================================
  datasets: {
    'general-chat': {
      name: 'General Chatbot',
      description: 'Multi-purpose conversational assistant',
      format: 'chat',
      categories: [
        { name: 'greetings', weight: 0.1, examples: 100 },
        { name: 'questions', weight: 0.3, examples: 300 },
        { name: 'tasks', weight: 0.3, examples: 300 },
        { name: 'creative', weight: 0.2, examples: 200 },
        { name: 'factual', weight: 0.1, examples: 100 }
      ],
      totalExamples: 1000,
      template: {
        system: 'You are a helpful, harmless, and honest assistant.',
        sampleConversations: [
          [
            { role: 'user', content: 'Hello!' },
            { role: 'assistant', content: 'Hello! How can I assist you today?' }
          ],
          [
            { role: 'user', content: 'What can you help me with?' },
            { role: 'assistant', content: 'I can help with questions, writing, coding, math, and more!' }
          ]
        ]
      }
    },

    'customer-support': {
      name: 'Customer Support Bot',
      description: 'Handle customer inquiries and issues',
      format: 'chat',
      categories: [
        { name: 'product_info', weight: 0.25 },
        { name: 'order_status', weight: 0.2 },
        { name: 'returns', weight: 0.15 },
        { name: 'technical_support', weight: 0.2 },
        { name: 'complaints', weight: 0.1 },
        { name: 'general', weight: 0.1 }
      ],
      template: {
        system: 'You are a helpful customer support agent for [COMPANY]. Be polite, professional, and solution-oriented.',
        placeholders: ['COMPANY', 'PRODUCT', 'POLICY'],
        sampleConversations: [
          [
            { role: 'user', content: 'Where is my order?' },
            { role: 'assistant', content: 'I\'d be happy to help track your order. Could you please provide your order number?' }
          ]
        ]
      }
    },

    'code-assistant': {
      name: 'Code Assistant',
      description: 'Programming help and code generation',
      format: 'instruction',
      categories: [
        { name: 'code_completion', weight: 0.25 },
        { name: 'bug_fixing', weight: 0.2 },
        { name: 'code_explanation', weight: 0.15 },
        { name: 'refactoring', weight: 0.15 },
        { name: 'documentation', weight: 0.1 },
        { name: 'testing', weight: 0.1 },
        { name: 'optimization', weight: 0.05 }
      ],
      languages: ['python', 'javascript', 'typescript', 'java', 'go', 'rust'],
      template: {
        instructionFormats: [
          'Write a function that {task}',
          'Fix the bug in this code: {code}',
          'Explain what this code does: {code}',
          'Refactor this code to be more efficient: {code}',
          'Add documentation to this function: {code}'
        ]
      }
    },

    'sql-generator': {
      name: 'SQL Generator',
      description: 'Natural language to SQL queries',
      format: 'instruction',
      template: {
        schemaFormat: `Table: {table_name}
Columns: {columns}
Relationships: {relationships}`,
        instructionFormats: [
          'Write a SQL query to {task}',
          'Given the schema, write SQL for: {query}',
          'Convert this to SQL: {natural_language}'
        ],
        sampleSchemas: [
          {
            tables: ['users', 'orders', 'products'],
            columns: {
              users: ['id', 'name', 'email', 'created_at'],
              orders: ['id', 'user_id', 'total', 'status'],
              products: ['id', 'name', 'price', 'category']
            }
          }
        ]
      }
    },

    'summarization': {
      name: 'Text Summarization',
      description: 'Condense long text into summaries',
      format: 'instruction',
      categories: [
        { name: 'news_articles', weight: 0.3 },
        { name: 'research_papers', weight: 0.2 },
        { name: 'meeting_notes', weight: 0.15 },
        { name: 'documents', weight: 0.2 },
        { name: 'conversations', weight: 0.15 }
      ],
      template: {
        instructionFormats: [
          'Summarize this text in {n} sentences: {text}',
          'Write a TL;DR for: {text}',
          'Extract the key points from: {text}',
          'Create a brief summary: {text}'
        ],
        lengthVariations: ['1 sentence', '2-3 sentences', 'a paragraph', 'bullet points']
      }
    }
  },

  // ============================================
  // DATA AUGMENTATION
  // ============================================
  augmentation: {
    'paraphrase': {
      name: 'Paraphrasing',
      description: 'Rephrase while preserving meaning',
      techniques: [
        'synonym_replacement',
        'sentence_restructuring',
        'active_passive_conversion',
        'formality_change'
      ],
      example: {
        original: 'The cat sat on the mat.',
        augmented: [
          'The feline rested on the rug.',
          'On the mat sat the cat.',
          'A cat was sitting on the mat.'
        ]
      }
    },

    'back_translation': {
      name: 'Back Translation',
      description: 'Translate to another language and back',
      languages: ['fr', 'de', 'es', 'zh', 'ja'],
      example: {
        original: 'Hello, how are you?',
        translated: 'Bonjour, comment allez-vous?',
        back: 'Hello, how are you doing?'
      }
    },

    'noise_injection': {
      name: 'Noise Injection',
      description: 'Add realistic errors for robustness',
      techniques: [
        { name: 'typos', rate: 0.01 },
        { name: 'missing_punctuation', rate: 0.05 },
        { name: 'case_errors', rate: 0.02 },
        { name: 'word_swap', rate: 0.01 }
      ]
    },

    'context_variation': {
      name: 'Context Variation',
      description: 'Vary the context while keeping core content',
      techniques: [
        'add_context',
        'remove_context',
        'change_perspective',
        'change_tone'
      ]
    }
  },

  // ============================================
  // TRAINING CONFIGS
  // ============================================
  training: {
    'pretrain-small': {
      name: 'Small Model Pretraining',
      modelSize: '< 100M params',
      config: {
        batch_size: 32,
        learning_rate: 3e-4,
        warmup_steps: 1000,
        max_steps: 100000,
        weight_decay: 0.1,
        gradient_accumulation: 4,
        fp16: true
      },
      hardware: 'Single GPU (16GB+)',
      estimatedTime: '1-2 days'
    },

    'finetune-lora': {
      name: 'LoRA Fine-tuning',
      description: 'Efficient fine-tuning with adapters',
      config: {
        lora_rank: 8,
        lora_alpha: 16,
        lora_dropout: 0.1,
        target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        batch_size: 4,
        learning_rate: 2e-4,
        epochs: 3,
        gradient_accumulation: 8
      },
      hardware: 'Single GPU (8GB+)',
      estimatedTime: '1-4 hours'
    },

    'finetune-full': {
      name: 'Full Fine-tuning',
      description: 'Update all model weights',
      config: {
        batch_size: 2,
        learning_rate: 1e-5,
        epochs: 1,
        warmup_ratio: 0.1,
        weight_decay: 0.01,
        gradient_checkpointing: true
      },
      hardware: 'Multi-GPU or high-memory GPU',
      estimatedTime: '4-24 hours'
    },

    'instruction-tuning': {
      name: 'Instruction Tuning',
      description: 'Train to follow instructions',
      config: {
        batch_size: 8,
        learning_rate: 2e-5,
        epochs: 3,
        max_seq_length: 2048,
        packing: true
      },
      dataRequirements: '10k-100k instruction examples'
    }
  }
}

/**
 * Generate sample dataset
 */
export function generateSampleDataset(template, count = 100) {
  const datasetTemplate = TrainingTemplates.datasets[template]
  if (!datasetTemplate) {
    throw new Error(`Unknown dataset template: ${template}`)
  }

  const samples = []
  const format = TrainingTemplates.formats[datasetTemplate.format]

  for (let i = 0; i < count; i++) {
    samples.push(generateSample(datasetTemplate, format, i))
  }

  return {
    format: datasetTemplate.format,
    samples,
    metadata: {
      template,
      count,
      generated: new Date().toISOString()
    }
  }
}

/**
 * Generate a single sample
 */
function generateSample(template, format, index) {
  // Placeholder - would be more sophisticated in practice
  if (format.name === 'Chat Format') {
    return {
      messages: [
        { role: 'system', content: template.template?.system || 'You are helpful.' },
        { role: 'user', content: `Sample question ${index}` },
        { role: 'assistant', content: `Sample answer ${index}` }
      ]
    }
  }

  if (format.name === 'Instruction Format') {
    return {
      instruction: `Sample instruction ${index}`,
      input: `Sample input ${index}`,
      output: `Sample output ${index}`
    }
  }

  return { text: `Sample text ${index}` }
}

/**
 * Validate dataset against format schema
 */
export function validateDataset(data, formatName) {
  const format = TrainingTemplates.formats[formatName]
  if (!format) {
    throw new Error(`Unknown format: ${formatName}`)
  }

  const errors = []
  const lines = data.trim().split('\n')

  lines.forEach((line, i) => {
    try {
      const obj = JSON.parse(line)
      const schemaErrors = validateAgainstSchema(obj, format.schema)
      if (schemaErrors.length > 0) {
        errors.push({ line: i + 1, errors: schemaErrors })
      }
    } catch (e) {
      errors.push({ line: i + 1, errors: [`Invalid JSON: ${e.message}`] })
    }
  })

  return {
    valid: errors.length === 0,
    totalLines: lines.length,
    errors
  }
}

function validateAgainstSchema(obj, schema) {
  const errors = []

  if (schema.required) {
    for (const field of schema.required) {
      if (!(field in obj)) {
        errors.push(`Missing required field: ${field}`)
      }
    }
  }

  return errors
}

/**
 * Convert between formats
 */
export function convertFormat(data, fromFormat, toFormat) {
  const lines = data.trim().split('\n')
  const converted = []

  for (const line of lines) {
    const obj = JSON.parse(line)
    converted.push(JSON.stringify(convertObject(obj, fromFormat, toFormat)))
  }

  return converted.join('\n')
}

function convertObject(obj, from, to) {
  // Chat → Instruction
  if (from === 'chat' && to === 'instruction') {
    const msgs = obj.messages.filter(m => m.role !== 'system')
    return {
      instruction: msgs[0]?.content || '',
      input: '',
      output: msgs[1]?.content || ''
    }
  }

  // Instruction → Chat
  if (from === 'instruction' && to === 'chat') {
    const messages = []
    if (obj.instruction) {
      let userMsg = obj.instruction
      if (obj.input) userMsg += '\n\n' + obj.input
      messages.push({ role: 'user', content: userMsg })
    }
    if (obj.output) {
      messages.push({ role: 'assistant', content: obj.output })
    }
    return { messages }
  }

  return obj
}

export default TrainingTemplates
