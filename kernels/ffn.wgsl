// Feed-Forward Network Kernel
// FFN(x) = GELU(xW1 + b1) * W2 + b2  (for SwiGLU/GeGLU variants)
// or FFN(x) = GELU(xW1 + b1)W2 + b2  (standard)

struct Params {
  batch_size: u32,
  seq_len: u32,
  hidden_dim: u32,
  intermediate_dim: u32,
}

// Standard FFN inputs
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;      // [hidden_dim, intermediate_dim]
@group(0) @binding(2) var<storage, read> b1: array<f32>;      // [intermediate_dim]
@group(0) @binding(3) var<storage, read> w2: array<f32>;      // [intermediate_dim, hidden_dim]
@group(0) @binding(4) var<storage, read> b2: array<f32>;      // [hidden_dim]
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> intermediate: array<f32, 4096>;  // Shared intermediate results

// GELU activation function (approximation)
fn gelu(x: f32) -> f32 {
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + tanh(inner));
}

// SiLU/Swish activation
fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn ffn_forward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let idx = global_id.x;
  let seq_pos = idx / params.hidden_dim;
  let hidden_idx = idx % params.hidden_dim;

  if (seq_pos >= params.seq_len) {
    return;
  }

  // Step 1: First linear layer (input -> intermediate)
  // This is computed per intermediate dimension
  let inter_idx = local_id.x % params.intermediate_dim;

  var sum1 = b1[inter_idx];
  for (var h = 0u; h < params.hidden_dim; h++) {
    let input_val = input[seq_pos * params.hidden_dim + h];
    let weight_val = w1[h * params.intermediate_dim + inter_idx];
    sum1 += input_val * weight_val;
  }

  // Apply GELU activation
  intermediate[local_id.x] = gelu(sum1);

  workgroupBarrier();

  // Step 2: Second linear layer (intermediate -> output)
  var sum2 = b2[hidden_idx];
  for (var i = 0u; i < params.intermediate_dim; i++) {
    let inter_val = intermediate[i];
    let weight_val = w2[i * params.hidden_dim + hidden_idx];
    sum2 += inter_val * weight_val;
  }

  output[idx] = sum2;
}

// SwiGLU variant: FFN(x) = SiLU(xW1) * (xV) @ W2
@group(1) @binding(0) var<storage, read> gate_weights: array<f32>;  // V matrix for gating

@compute @workgroup_size(256)
fn swiglu_forward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let idx = global_id.x;
  let seq_pos = idx / params.hidden_dim;
  let hidden_idx = idx % params.hidden_dim;

  if (seq_pos >= params.seq_len) {
    return;
  }

  let inter_idx = local_id.x % params.intermediate_dim;

  // Compute gate and up projection
  var gate_sum = 0.0;
  var up_sum = 0.0;

  for (var h = 0u; h < params.hidden_dim; h++) {
    let input_val = input[seq_pos * params.hidden_dim + h];
    gate_sum += input_val * w1[h * params.intermediate_dim + inter_idx];
    up_sum += input_val * gate_weights[h * params.intermediate_dim + inter_idx];
  }

  // SwiGLU: SiLU(gate) * up
  intermediate[local_id.x] = silu(gate_sum) * up_sum;

  workgroupBarrier();

  // Down projection
  var sum2 = b2[hidden_idx];
  for (var i = 0u; i < params.intermediate_dim; i++) {
    sum2 += intermediate[i] * w2[i * params.hidden_dim + hidden_idx];
  }

  output[idx] = sum2;
}
