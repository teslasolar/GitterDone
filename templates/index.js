/**
 * Templates Index
 * Central export for all template types
 */

export { ModelTemplates, getAllTemplates, getTemplate, getTemplatesByCategory, filterTemplates } from './models.js'
export { DeploymentTemplates, generateDeployment, recommendDeployment } from './deploy.js'
export { TrainingTemplates, generateSampleDataset, validateDataset, convertFormat } from './training.js'

/**
 * Quick Start Templates
 * Common configurations for rapid prototyping
 */
export const QuickStart = {
  /**
   * Minimal chatbot setup
   */
  chatbot: {
    model: 'chat-small',
    deployment: 'github-pages',
    training: 'general-chat',
    description: 'Simple conversational AI'
  },

  /**
   * Code assistant setup
   */
  codeAssistant: {
    model: 'code-small',
    deployment: 'vercel',
    training: 'code-assistant',
    description: 'Programming helper'
  },

  /**
   * Customer support bot
   */
  supportBot: {
    model: 'chat-medium',
    deployment: 'docker',
    training: 'customer-support',
    description: 'Customer service automation'
  },

  /**
   * Minimal demo setup
   */
  demo: {
    model: 'nano-gpt',
    deployment: 'github-pages',
    training: null,
    description: 'Lightweight demo/learning'
  },

  /**
   * Enterprise setup
   */
  enterprise: {
    model: 'medium-llm',
    deployment: 'aws-s3',
    training: 'finetune-lora',
    description: 'Production-grade deployment'
  }
}

/**
 * Get complete setup for a use case
 */
export function getQuickStartSetup(useCase) {
  const setup = QuickStart[useCase]
  if (!setup) return null

  const { ModelTemplates } = require('./models.js')
  const { DeploymentTemplates } = require('./deploy.js')
  const { TrainingTemplates } = require('./training.js')

  // Find the model template
  let modelTemplate = null
  for (const category of Object.values(ModelTemplates)) {
    if (category[setup.model]) {
      modelTemplate = category[setup.model]
      break
    }
  }

  return {
    useCase,
    description: setup.description,
    model: modelTemplate,
    deployment: DeploymentTemplates[setup.deployment],
    training: setup.training ? TrainingTemplates.datasets[setup.training] : null
  }
}

/**
 * List all available templates
 */
export function listAllTemplates() {
  const { ModelTemplates } = require('./models.js')
  const { DeploymentTemplates } = require('./deploy.js')
  const { TrainingTemplates } = require('./training.js')

  const models = []
  for (const [category, items] of Object.entries(ModelTemplates)) {
    for (const [id, template] of Object.entries(items)) {
      models.push({ id, category, ...template })
    }
  }

  const deployments = Object.entries(DeploymentTemplates).map(([id, t]) => ({
    id,
    name: t.name,
    platform: t.platform,
    description: t.description
  }))

  const datasets = Object.entries(TrainingTemplates.datasets).map(([id, t]) => ({
    id,
    name: t.name,
    format: t.format,
    description: t.description
  }))

  const formats = Object.entries(TrainingTemplates.formats).map(([id, t]) => ({
    id,
    name: t.name,
    description: t.description
  }))

  return { models, deployments, datasets, formats, quickStart: Object.keys(QuickStart) }
}

export default {
  QuickStart,
  getQuickStartSetup,
  listAllTemplates
}
