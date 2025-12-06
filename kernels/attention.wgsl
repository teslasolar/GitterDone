// Self-Attention Kernel
// Computes scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V

struct Params {
  seq_len: u32,      // Sequence length
  head_dim: u32,     // Dimension per head
  num_heads: u32,    // Number of attention heads
  scale: f32,        // 1.0 / sqrt(head_dim)
}

// Input tensors
@group(0) @binding(0) var<storage, read> query: array<f32>;     // [seq_len, num_heads, head_dim]
@group(0) @binding(1) var<storage, read> key: array<f32>;       // [seq_len, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> value: array<f32>;     // [seq_len, num_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [seq_len, num_heads, head_dim]
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for attention scores
var<workgroup> attention_scores: array<f32, 1024>;  // Max seq_len^2 = 32^2 per head
var<workgroup> max_score: array<f32, 32>;           // For softmax stability
var<workgroup> sum_exp: array<f32, 32>;             // For softmax normalization

@compute @workgroup_size(32, 1)
fn compute_attention(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let seq_idx = global_id.x;
  let head_idx = global_id.y;
  let local_seq = local_id.x;

  if (seq_idx >= params.seq_len || head_idx >= params.num_heads) {
    return;
  }

  let head_offset = head_idx * params.head_dim;

  // Step 1: Compute Q @ K^T scores for this query position
  for (var k = 0u; k < params.seq_len; k++) {
    var score = 0.0;
    for (var d = 0u; d < params.head_dim; d++) {
      let q_val = query[seq_idx * params.num_heads * params.head_dim + head_offset + d];
      let k_val = key[k * params.num_heads * params.head_dim + head_offset + d];
      score += q_val * k_val;
    }
    attention_scores[local_seq * params.seq_len + k] = score * params.scale;
  }

  workgroupBarrier();

  // Step 2: Find max for numerical stability
  var local_max = -1e10;
  for (var k = 0u; k < params.seq_len; k++) {
    let score = attention_scores[local_seq * params.seq_len + k];
    local_max = max(local_max, score);
  }
  max_score[local_seq] = local_max;

  workgroupBarrier();

  // Step 3: Compute exp and sum
  var local_sum = 0.0;
  for (var k = 0u; k < params.seq_len; k++) {
    let score = attention_scores[local_seq * params.seq_len + k];
    let exp_score = exp(score - max_score[local_seq]);
    attention_scores[local_seq * params.seq_len + k] = exp_score;
    local_sum += exp_score;
  }
  sum_exp[local_seq] = local_sum;

  workgroupBarrier();

  // Step 4: Normalize (softmax) and compute output
  for (var d = 0u; d < params.head_dim; d++) {
    var weighted_sum = 0.0;
    for (var k = 0u; k < params.seq_len; k++) {
      let attention_weight = attention_scores[local_seq * params.seq_len + k] / sum_exp[local_seq];
      let v_val = value[k * params.num_heads * params.head_dim + head_offset + d];
      weighted_sum += attention_weight * v_val;
    }
    output[seq_idx * params.num_heads * params.head_dim + head_offset + d] = weighted_sum;
  }
}

// Causal (masked) attention variant
@compute @workgroup_size(32, 1)
fn compute_causal_attention(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let seq_idx = global_id.x;
  let head_idx = global_id.y;
  let local_seq = local_id.x;

  if (seq_idx >= params.seq_len || head_idx >= params.num_heads) {
    return;
  }

  let head_offset = head_idx * params.head_dim;

  // Compute scores with causal mask
  for (var k = 0u; k < params.seq_len; k++) {
    if (k > seq_idx) {
      // Mask future positions
      attention_scores[local_seq * params.seq_len + k] = -1e10;
    } else {
      var score = 0.0;
      for (var d = 0u; d < params.head_dim; d++) {
        let q_val = query[seq_idx * params.num_heads * params.head_dim + head_offset + d];
        let k_val = key[k * params.num_heads * params.head_dim + head_offset + d];
        score += q_val * k_val;
      }
      attention_scores[local_seq * params.seq_len + k] = score * params.scale;
    }
  }

  workgroupBarrier();

  // Softmax and output (same as above)
  var local_max = -1e10;
  for (var k = 0u; k <= seq_idx; k++) {
    let score = attention_scores[local_seq * params.seq_len + k];
    local_max = max(local_max, score);
  }

  var local_sum = 0.0;
  for (var k = 0u; k <= seq_idx; k++) {
    let score = attention_scores[local_seq * params.seq_len + k];
    let exp_score = exp(score - local_max);
    attention_scores[local_seq * params.seq_len + k] = exp_score;
    local_sum += exp_score;
  }

  for (var d = 0u; d < params.head_dim; d++) {
    var weighted_sum = 0.0;
    for (var k = 0u; k <= seq_idx; k++) {
      let attention_weight = attention_scores[local_seq * params.seq_len + k] / local_sum;
      let v_val = value[k * params.num_heads * params.head_dim + head_offset + d];
      weighted_sum += attention_weight * v_val;
    }
    output[seq_idx * params.num_heads * params.head_dim + head_offset + d] = weighted_sum;
  }
}
