/**
 * MCP Agent System
 *
 * Orchestrates LLM interactions with MCP tools for autonomous task execution
 */

import { MCPClient } from './protocol.js'
import { registerBuiltInTools } from './tools.js'

export class Agent {
  constructor(pipeline, options = {}) {
    this.pipeline = pipeline
    this.mcp = new MCPClient(options.mcp || {})

    this.options = {
      maxIterations: options.maxIterations || 10,
      maxToolCalls: options.maxToolCalls || 50,
      temperature: options.temperature || 0.7,
      systemPrompt: options.systemPrompt || DEFAULT_SYSTEM_PROMPT,
      toolCallFormat: options.toolCallFormat || 'json',
      verbose: options.verbose || false,
      ...options
    }

    this.context = {
      messages: [],
      toolCalls: [],
      iterations: 0,
      totalToolCalls: 0
    }

    this.eventHandlers = new Map()

    // Register built-in tools
    registerBuiltInTools(this.mcp)
  }

  /**
   * Connect to MCP servers
   */
  async connect(servers) {
    for (const server of servers) {
      await this.mcp.connect(server)
    }
    return this
  }

  /**
   * Run the agent on a task
   */
  async run(task, options = {}) {
    const opts = { ...this.options, ...options }

    // Reset context
    this.context = {
      messages: [],
      toolCalls: [],
      iterations: 0,
      totalToolCalls: 0
    }

    // Add system prompt with tool descriptions
    const toolsDescription = this.formatToolsForPrompt()
    const systemPrompt = opts.systemPrompt.replace('{tools}', toolsDescription)

    this.context.messages.push({
      role: 'system',
      content: systemPrompt
    })

    // Add user task
    this.context.messages.push({
      role: 'user',
      content: task
    })

    this.emit('start', { task })

    // Agent loop
    while (this.context.iterations < opts.maxIterations) {
      this.context.iterations++
      this.emit('iteration', { iteration: this.context.iterations })

      // Generate response
      const prompt = this.formatMessages()
      let response = ''

      for await (const token of this.pipeline.generate(prompt, {
        temperature: opts.temperature,
        maxTokens: opts.maxTokens || 2048,
        stopSequences: ['</tool_call>', '</final_answer>']
      })) {
        response += token
        this.emit('token', { token })
      }

      this.emit('response', { response })

      // Parse response for tool calls
      const toolCalls = this.parseToolCalls(response)

      if (toolCalls.length > 0) {
        // Execute tool calls
        for (const call of toolCalls) {
          if (this.context.totalToolCalls >= opts.maxToolCalls) {
            this.emit('error', { error: 'Max tool calls exceeded' })
            break
          }

          this.emit('tool:calling', call)

          try {
            const result = await this.executeTool(call)
            this.context.toolCalls.push({ call, result })
            this.context.totalToolCalls++

            // Add to messages
            this.context.messages.push({
              role: 'assistant',
              content: response
            })

            this.context.messages.push({
              role: 'tool',
              name: call.name,
              content: JSON.stringify(result)
            })

            this.emit('tool:result', { call, result })
          } catch (error) {
            this.emit('tool:error', { call, error })

            this.context.messages.push({
              role: 'tool',
              name: call.name,
              content: JSON.stringify({ error: error.message })
            })
          }
        }
      } else {
        // Check for final answer
        const finalAnswer = this.parseFinalAnswer(response)

        if (finalAnswer) {
          this.context.messages.push({
            role: 'assistant',
            content: response
          })

          this.emit('complete', {
            answer: finalAnswer,
            iterations: this.context.iterations,
            toolCalls: this.context.totalToolCalls
          })

          return {
            success: true,
            answer: finalAnswer,
            context: this.context
          }
        }

        // No tool call and no final answer - assume response is complete
        this.context.messages.push({
          role: 'assistant',
          content: response
        })

        this.emit('complete', {
          answer: response,
          iterations: this.context.iterations,
          toolCalls: this.context.totalToolCalls
        })

        return {
          success: true,
          answer: response,
          context: this.context
        }
      }
    }

    // Max iterations reached
    this.emit('timeout', { iterations: this.context.iterations })

    return {
      success: false,
      error: 'Max iterations reached',
      context: this.context
    }
  }

  /**
   * Format tools for system prompt
   */
  formatToolsForPrompt() {
    const tools = this.mcp.listTools()

    return tools.map(tool => {
      const params = tool.inputSchema?.properties || {}
      const required = tool.inputSchema?.required || []

      const paramStr = Object.entries(params)
        .map(([name, schema]) => {
          const req = required.includes(name) ? ' (required)' : ''
          return `  - ${name}: ${schema.type || 'any'}${req} - ${schema.description || ''}`
        })
        .join('\n')

      return `
## ${tool.name}
${tool.description}

Parameters:
${paramStr || '  (none)'}
`
    }).join('\n')
  }

  /**
   * Format messages for prompt
   */
  formatMessages() {
    return this.context.messages.map(msg => {
      switch (msg.role) {
        case 'system':
          return `<system>\n${msg.content}\n</system>`
        case 'user':
          return `<user>\n${msg.content}\n</user>`
        case 'assistant':
          return `<assistant>\n${msg.content}\n</assistant>`
        case 'tool':
          return `<tool_result name="${msg.name}">\n${msg.content}\n</tool_result>`
        default:
          return msg.content
      }
    }).join('\n\n')
  }

  /**
   * Parse tool calls from response
   */
  parseToolCalls(response) {
    const calls = []

    // JSON format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    const jsonPattern = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g
    let match

    while ((match = jsonPattern.exec(response)) !== null) {
      try {
        const parsed = JSON.parse(match[1])
        calls.push({
          name: parsed.name || parsed.tool,
          arguments: parsed.arguments || parsed.args || parsed.parameters || {}
        })
      } catch (e) {
        // Invalid JSON, skip
        if (this.options.verbose) {
          console.warn('Failed to parse tool call:', match[1])
        }
      }
    }

    // Also check for function-call style
    const funcPattern = /(\w+)\(([\s\S]*?)\)/g
    if (calls.length === 0) {
      while ((match = funcPattern.exec(response)) !== null) {
        const toolName = match[1]
        if (this.mcp.tools.has(`local:${toolName}`)) {
          try {
            const args = match[2] ? JSON.parse(`{${match[2]}}`) : {}
            calls.push({ name: toolName, arguments: args })
          } catch (e) {
            // Not valid args
          }
        }
      }
    }

    return calls
  }

  /**
   * Parse final answer from response
   */
  parseFinalAnswer(response) {
    const pattern = /<final_answer>\s*([\s\S]*?)\s*<\/final_answer>/
    const match = response.match(pattern)
    return match ? match[1].trim() : null
  }

  /**
   * Execute a tool call
   */
  async executeTool(call) {
    const { name, arguments: args } = call

    // Check for local tool first
    const localTool = this.mcp.tools.get(`local:${name}`)
    if (localTool && localTool.handler) {
      return await localTool.handler(args)
    }

    // Try MCP tool
    return await this.mcp.callTool(name, args)
  }

  /**
   * Add a custom tool
   */
  addTool(tool) {
    this.mcp.registerLocalTool(tool)
    return this
  }

  /**
   * Event handling
   */
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, [])
    }
    this.eventHandlers.get(event).push(handler)
    return this
  }

  emit(event, data) {
    const handlers = this.eventHandlers.get(event) || []
    for (const handler of handlers) {
      handler(data)
    }
  }

  /**
   * Get conversation history
   */
  getHistory() {
    return [...this.context.messages]
  }

  /**
   * Clear context
   */
  clear() {
    this.context = {
      messages: [],
      toolCalls: [],
      iterations: 0,
      totalToolCalls: 0
    }
    return this
  }

  /**
   * Destroy agent
   */
  destroy() {
    this.mcp.disconnectAll()
    this.clear()
  }
}

/**
 * Default system prompt for agents
 */
const DEFAULT_SYSTEM_PROMPT = `You are an AI assistant with access to tools. Use tools when needed to complete tasks.

# Available Tools
{tools}

# How to Use Tools
To call a tool, use this format:
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Wait for the tool result before continuing.

# Completing Tasks
When you have completed the task, provide your final answer:
<final_answer>
Your final response here
</final_answer>

# Guidelines
- Think step by step
- Use tools when you need external information or actions
- Explain your reasoning
- If a tool fails, try alternative approaches
- Be concise but thorough`

/**
 * Create an agent with sensible defaults
 */
export function createAgent(pipeline, options = {}) {
  return new Agent(pipeline, options)
}

/**
 * ReAct (Reasoning + Acting) Agent
 */
export class ReActAgent extends Agent {
  constructor(pipeline, options = {}) {
    super(pipeline, {
      ...options,
      systemPrompt: options.systemPrompt || REACT_SYSTEM_PROMPT
    })
  }

  parseToolCalls(response) {
    // ReAct format: Action: tool_name[arg1, arg2]
    const calls = []

    const pattern = /Action:\s*(\w+)\[(.*?)\]/g
    let match

    while ((match = pattern.exec(response)) !== null) {
      const name = match[1]
      const argsStr = match[2]

      // Parse arguments
      let args = {}
      if (argsStr) {
        try {
          args = JSON.parse(`{${argsStr}}`)
        } catch {
          // Simple comma-separated args
          const parts = argsStr.split(',').map(s => s.trim())
          const tool = this.mcp.tools.get(`local:${name}`)
          if (tool?.inputSchema?.properties) {
            const paramNames = Object.keys(tool.inputSchema.properties)
            parts.forEach((val, i) => {
              if (paramNames[i]) {
                args[paramNames[i]] = val.replace(/^["']|["']$/g, '')
              }
            })
          }
        }
      }

      calls.push({ name, arguments: args })
    }

    // Fall back to JSON format
    if (calls.length === 0) {
      return super.parseToolCalls(response)
    }

    return calls
  }

  parseFinalAnswer(response) {
    // ReAct format: Final Answer: ...
    const pattern = /Final Answer:\s*([\s\S]*?)(?=$|\nThought:|\nAction:)/
    const match = response.match(pattern)
    if (match) return match[1].trim()

    return super.parseFinalAnswer(response)
  }
}

const REACT_SYSTEM_PROMPT = `You are an AI assistant that solves problems step by step using the ReAct framework.

# Available Tools
{tools}

# ReAct Format
Always follow this format:

Thought: Think about what to do next
Action: tool_name[argument1, argument2]
Observation: (tool result will appear here)
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Final Answer: Your final response

# Guidelines
- Always start with a Thought
- Use tools to gather information
- After each Observation, reflect on what you learned
- Be methodical and thorough
- If one approach fails, try another`

/**
 * Plan-and-Execute Agent
 */
export class PlanExecuteAgent extends Agent {
  constructor(pipeline, options = {}) {
    super(pipeline, {
      ...options,
      systemPrompt: options.systemPrompt || PLAN_EXECUTE_PROMPT
    })
    this.plan = []
    this.currentStep = 0
  }

  async run(task, options = {}) {
    this.emit('planning', { task })

    // First, create a plan
    const planPrompt = `Create a step-by-step plan to accomplish this task:

Task: ${task}

Available tools:
${this.formatToolsForPrompt()}

Respond with a numbered list of steps.`

    let planResponse = ''
    for await (const token of this.pipeline.generate(planPrompt, {
      temperature: 0.3,
      maxTokens: 1024
    })) {
      planResponse += token
    }

    // Parse plan
    this.plan = planResponse.split('\n')
      .filter(line => /^\d+\./.test(line.trim()))
      .map(line => line.replace(/^\d+\.\s*/, '').trim())

    this.emit('plan:created', { plan: this.plan })

    // Execute plan step by step
    const results = []

    for (let i = 0; i < this.plan.length; i++) {
      this.currentStep = i
      const step = this.plan[i]

      this.emit('step:start', { step: i + 1, description: step })

      // Execute step as a sub-task
      const stepResult = await super.run(`Execute this step: ${step}\n\nContext from previous steps:\n${results.map((r, j) => `Step ${j + 1}: ${r}`).join('\n')}`, {
        ...options,
        maxIterations: 5
      })

      results.push(stepResult.answer || stepResult.error || 'Completed')

      this.emit('step:complete', { step: i + 1, result: stepResult })

      if (!stepResult.success) {
        break
      }
    }

    // Synthesize final answer
    const synthesisPrompt = `Task: ${task}

Completed steps and results:
${this.plan.map((step, i) => `${i + 1}. ${step}\n   Result: ${results[i] || 'Not completed'}`).join('\n')}

Provide a final summary and answer.`

    let finalAnswer = ''
    for await (const token of this.pipeline.generate(synthesisPrompt, {
      temperature: 0.5,
      maxTokens: 1024
    })) {
      finalAnswer += token
    }

    this.emit('complete', { answer: finalAnswer, plan: this.plan, results })

    return {
      success: true,
      answer: finalAnswer,
      plan: this.plan,
      results,
      context: this.context
    }
  }
}

const PLAN_EXECUTE_PROMPT = `You are a task execution agent. Execute the given step using available tools.

# Available Tools
{tools}

# Instructions
1. Focus only on the current step
2. Use tools as needed
3. Report the result clearly

# Tool Calling
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

# Completion
<final_answer>
Step result here
</final_answer>`

export default Agent
