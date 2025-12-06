/**
 * ModelSpace - 3D Navigation through Model Layers
 * Allows visual exploration of model weights in a 3D tensor space
 */
export class ModelSpace {
  constructor(model, canvas) {
    this.model = model
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.layers = model.layers || []

    // Navigation state
    this.pos = {
      layer: 0,    // Current layer index
      row: 0,      // Row in weight matrix
      col: 0,      // Column in weight matrix
      head: 0      // Attention head (for attention layers)
    }

    this.zoom = 1.0
    this.offset = { x: 0, y: 0 }

    // Visualization settings
    this.colorMap = 'green'  // 'green', 'viridis', 'plasma'
    this.showGrid = true
    this.showLabels = true

    // Cached data
    this.currentWeights = null
    this.currentShape = null

    this.setupControls()
  }

  /**
   * Setup keyboard controls
   */
  setupControls() {
    document.addEventListener('keydown', (e) => {
      switch (e.key.toLowerCase()) {
        case 'w': this.navigate(0, -1, 0); break
        case 's': this.navigate(0, 1, 0); break
        case 'a': this.navigate(0, 0, -1); break
        case 'd': this.navigate(0, 0, 1); break
        case 'q': this.navigate(-1, 0, 0); break
        case 'e': this.navigate(1, 0, 0); break
        case 'r': this.navigateHead(1); break
        case 'f': this.navigateHead(-1); break
        case '+': case '=': this.setZoom(this.zoom * 1.2); break
        case '-': this.setZoom(this.zoom / 1.2); break
        case '0': this.resetView(); break
      }
    })

    // Mouse wheel zoom
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault()
      const factor = e.deltaY > 0 ? 0.9 : 1.1
      this.setZoom(this.zoom * factor)
    })

    // Mouse drag for panning
    let dragging = false
    let lastPos = { x: 0, y: 0 }

    this.canvas.addEventListener('mousedown', (e) => {
      dragging = true
      lastPos = { x: e.clientX, y: e.clientY }
    })

    this.canvas.addEventListener('mousemove', (e) => {
      if (!dragging) return
      this.offset.x += e.clientX - lastPos.x
      this.offset.y += e.clientY - lastPos.y
      lastPos = { x: e.clientX, y: e.clientY }
      this.render()
    })

    this.canvas.addEventListener('mouseup', () => dragging = false)
    this.canvas.addEventListener('mouseleave', () => dragging = false)
  }

  /**
   * Navigate through tensor space
   */
  navigate(dLayer, dRow, dCol) {
    const layer = this.layers[this.pos.layer]
    if (!layer) return

    // Update position with bounds checking
    this.pos.layer = Math.max(0, Math.min(this.layers.length - 1, this.pos.layer + dLayer))

    if (this.currentShape) {
      const maxRow = this.currentShape[0] || 1
      const maxCol = this.currentShape[1] || 1
      this.pos.row = Math.max(0, Math.min(maxRow - 1, this.pos.row + dRow))
      this.pos.col = Math.max(0, Math.min(maxCol - 1, this.pos.col + dCol))
    }

    this.loadAndRender()
  }

  /**
   * Navigate attention heads
   */
  navigateHead(delta) {
    const layer = this.layers[this.pos.layer]
    if (!layer || layer.type !== 'attention') return

    const numHeads = layer.params?.num_heads || 1
    this.pos.head = Math.max(0, Math.min(numHeads - 1, this.pos.head + delta))
    this.loadAndRender()
  }

  /**
   * Set zoom level
   */
  setZoom(zoom) {
    this.zoom = Math.max(0.1, Math.min(10, zoom))
    this.render()
  }

  /**
   * Reset view
   */
  resetView() {
    this.zoom = 1.0
    this.offset = { x: 0, y: 0 }
    this.render()
  }

  /**
   * Load weights and render current view
   */
  async loadAndRender() {
    const layer = this.layers[this.pos.layer]
    if (!layer) return

    try {
      // Load layer weights if not already loaded
      if (this.model.loadLayerWeights) {
        await this.model.loadLayerWeights(layer)
      }

      // Get weight data
      if (layer.weights && this.model.tensorCube) {
        this.currentWeights = await this.model.tensorCube.readTensor(layer.weights.name)
        this.currentShape = layer.weights.shape || [Math.sqrt(this.currentWeights.length) | 0]
      } else {
        this.currentWeights = null
        this.currentShape = null
      }

      this.render()
    } catch (error) {
      console.error('Failed to load layer weights:', error)
      this.renderError(error.message)
    }
  }

  /**
   * Main render function
   */
  render() {
    const { width, height } = this.canvas
    this.ctx.fillStyle = '#000'
    this.ctx.fillRect(0, 0, width, height)

    if (!this.currentWeights) {
      this.renderPlaceholder()
      return
    }

    // Draw weight matrix
    this.drawWeightMatrix()

    // Draw UI overlays
    if (this.showGrid) this.drawGrid()
    if (this.showLabels) this.drawLabels()
    this.drawNavigationInfo()
  }

  /**
   * Draw weight matrix visualization
   */
  drawWeightMatrix() {
    const { width, height } = this.canvas
    const data = this.currentWeights
    const shape = this.currentShape

    // Determine matrix dimensions
    let rows, cols
    if (shape.length >= 2) {
      rows = shape[0]
      cols = shape[1]
    } else {
      // 1D tensor - reshape to square-ish
      const total = data.length
      cols = Math.ceil(Math.sqrt(total))
      rows = Math.ceil(total / cols)
    }

    // Calculate cell size
    const cellW = (width * this.zoom) / cols
    const cellH = (height * this.zoom) / rows

    // Find value range for normalization
    let min = Infinity, max = -Infinity
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i]
      if (data[i] > max) max = data[i]
    }
    const range = max - min || 1

    // Draw cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c
        if (idx >= data.length) continue

        const val = (data[idx] - min) / range
        const x = c * cellW + this.offset.x
        const y = r * cellH + this.offset.y

        // Skip if outside visible area
        if (x + cellW < 0 || x > width || y + cellH < 0 || y > height) continue

        this.ctx.fillStyle = this.valueToColor(val)
        this.ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5)
      }
    }

    // Highlight current position
    const curX = this.pos.col * cellW + this.offset.x
    const curY = this.pos.row * cellH + this.offset.y
    this.ctx.strokeStyle = '#fff'
    this.ctx.lineWidth = 2
    this.ctx.strokeRect(curX, curY, cellW, cellH)
  }

  /**
   * Convert value to color
   */
  valueToColor(val) {
    val = Math.max(0, Math.min(1, val))

    switch (this.colorMap) {
      case 'viridis':
        return this.viridis(val)
      case 'plasma':
        return this.plasma(val)
      default:
        // Green gradient
        const g = Math.floor(val * 255)
        return `rgb(0, ${g}, 0)`
    }
  }

  viridis(t) {
    // Simplified viridis colormap
    const r = Math.floor(68 + t * (253 - 68))
    const g = Math.floor(1 + t * (231 - 1))
    const b = Math.floor(84 + (1 - t) * (36))
    return `rgb(${r}, ${g}, ${b})`
  }

  plasma(t) {
    // Simplified plasma colormap
    const r = Math.floor(13 + t * (240 - 13))
    const g = Math.floor(8 + Math.sin(t * Math.PI) * 150)
    const b = Math.floor(135 + (1 - t) * (100))
    return `rgb(${r}, ${g}, ${b})`
  }

  /**
   * Draw grid overlay
   */
  drawGrid() {
    const { width, height } = this.canvas

    this.ctx.strokeStyle = '#333'
    this.ctx.lineWidth = 0.5

    const gridSize = 50 * this.zoom

    for (let x = this.offset.x % gridSize; x < width; x += gridSize) {
      this.ctx.beginPath()
      this.ctx.moveTo(x, 0)
      this.ctx.lineTo(x, height)
      this.ctx.stroke()
    }

    for (let y = this.offset.y % gridSize; y < height; y += gridSize) {
      this.ctx.beginPath()
      this.ctx.moveTo(0, y)
      this.ctx.lineTo(width, y)
      this.ctx.stroke()
    }
  }

  /**
   * Draw labels
   */
  drawLabels() {
    const layer = this.layers[this.pos.layer]
    if (!layer) return

    this.ctx.fillStyle = '#0f0'
    this.ctx.font = '12px monospace'

    // Layer name
    this.ctx.fillText(`Layer: ${layer.name}`, 10, 20)
    this.ctx.fillText(`Type: ${layer.type}`, 10, 35)

    if (this.currentShape) {
      this.ctx.fillText(`Shape: [${this.currentShape.join(', ')}]`, 10, 50)
    }
  }

  /**
   * Draw navigation info
   */
  drawNavigationInfo() {
    const { width, height } = this.canvas

    this.ctx.fillStyle = '#0f0'
    this.ctx.font = '10px monospace'

    const nav = [
      `Layer ${this.pos.layer + 1}/${this.layers.length}`,
      `Row: ${this.pos.row}`,
      `Col: ${this.pos.col}`,
      `Zoom: ${this.zoom.toFixed(1)}x`
    ]

    this.ctx.fillText(nav.join(' | '), 10, height - 10)

    // Controls hint
    this.ctx.fillStyle = '#666'
    this.ctx.fillText('WASD: navigate | QE: layers | RF: heads | +/-: zoom', width - 300, height - 10)
  }

  /**
   * Render placeholder when no weights available
   */
  renderPlaceholder() {
    const { width, height } = this.canvas

    // Draw grid pattern
    for (let y = 0; y < height; y += 10) {
      for (let x = 0; x < width; x += 10) {
        const val = Math.sin(x * 0.05) * Math.cos(y * 0.05) * 0.5 + 0.5
        this.ctx.fillStyle = `rgb(0, ${Math.floor(val * 100)}, 0)`
        this.ctx.fillRect(x, y, 9, 9)
      }
    }

    this.ctx.fillStyle = '#0f0'
    this.ctx.font = '16px monospace'
    this.ctx.textAlign = 'center'
    this.ctx.fillText('No weights loaded', width / 2, height / 2)
    this.ctx.fillText('Press Q/E to change layers', width / 2, height / 2 + 20)
    this.ctx.textAlign = 'left'
  }

  /**
   * Render error state
   */
  renderError(message) {
    const { width, height } = this.canvas

    this.ctx.fillStyle = '#000'
    this.ctx.fillRect(0, 0, width, height)

    this.ctx.fillStyle = '#f00'
    this.ctx.font = '14px monospace'
    this.ctx.textAlign = 'center'
    this.ctx.fillText('Error: ' + message, width / 2, height / 2)
    this.ctx.textAlign = 'left'
  }

  /**
   * Get current position info
   */
  getPosition() {
    const layer = this.layers[this.pos.layer]
    return {
      layer: this.pos.layer,
      layerName: layer?.name || 'unknown',
      layerType: layer?.type || 'unknown',
      row: this.pos.row,
      col: this.pos.col,
      head: this.pos.head,
      zoom: this.zoom,
      shape: this.currentShape
    }
  }

  /**
   * Jump to specific position
   */
  goTo(layer, row = 0, col = 0) {
    this.pos.layer = Math.max(0, Math.min(this.layers.length - 1, layer))
    this.pos.row = row
    this.pos.col = col
    this.loadAndRender()
  }

  /**
   * Set color map
   */
  setColorMap(colorMap) {
    this.colorMap = colorMap
    this.render()
  }
}

export default ModelSpace
