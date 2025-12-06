// Embedding Lookup Kernel
// Converts token IDs to embedding vectors

struct Params {
  vocab_size: u32,     // Size of vocabulary
  embed_dim: u32,      // Embedding dimension
  seq_len: u32,        // Sequence length
  max_seq_len: u32,    // Maximum sequence length for positional embeddings
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> token_embeddings: array<f32>;  // [vocab_size, embed_dim]
@group(0) @binding(2) var<storage, read> pos_embeddings: array<f32>;    // [max_seq_len, embed_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;       // [seq_len, embed_dim]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn embed_tokens(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let idx = global_id.x;
  let seq_pos = idx / params.embed_dim;
  let embed_idx = idx % params.embed_dim;

  if (seq_pos >= params.seq_len) {
    return;
  }

  // Get token ID for this position
  let token_id = token_ids[seq_pos];

  // Bounds check
  if (token_id >= params.vocab_size) {
    output[idx] = 0.0;
    return;
  }

  // Lookup token embedding
  let token_embed_offset = token_id * params.embed_dim + embed_idx;
  let token_embed = token_embeddings[token_embed_offset];

  // Add positional embedding
  let pos_embed_offset = seq_pos * params.embed_dim + embed_idx;
  let pos_embed = pos_embeddings[pos_embed_offset];

  output[idx] = token_embed + pos_embed;
}

// Rotary Position Embedding (RoPE)
@compute @workgroup_size(256)
fn apply_rope(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let idx = global_id.x;
  let seq_pos = idx / params.embed_dim;
  let embed_idx = idx % params.embed_dim;
  let half_dim = params.embed_dim / 2u;

  if (seq_pos >= params.seq_len) {
    return;
  }

  // Compute rotation frequency
  let freq_idx = embed_idx % half_dim;
  let theta = pow(10000.0, -2.0 * f32(freq_idx) / f32(params.embed_dim));
  let angle = f32(seq_pos) * theta;

  let cos_val = cos(angle);
  let sin_val = sin(angle);

  // Apply rotation
  let base_idx = seq_pos * params.embed_dim;
  if (embed_idx < half_dim) {
    // First half
    let x0 = output[base_idx + embed_idx];
    let x1 = output[base_idx + embed_idx + half_dim];
    output[base_idx + embed_idx] = x0 * cos_val - x1 * sin_val;
  } else {
    // Second half
    let x0 = output[base_idx + embed_idx - half_dim];
    let x1 = output[base_idx + embed_idx];
    output[base_idx + embed_idx] = x0 * sin_val + x1 * cos_val;
  }
}

// Sinusoidal positional encoding (Transformer style)
@compute @workgroup_size(256)
fn sinusoidal_pos_encoding(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let idx = global_id.x;
  let pos = idx / params.embed_dim;
  let i = idx % params.embed_dim;

  if (pos >= params.max_seq_len) {
    return;
  }

  let half_dim = params.embed_dim / 2u;
  let dim_idx = i % half_dim;
  let freq = pow(10000.0, -2.0 * f32(dim_idx) / f32(params.embed_dim));
  let angle = f32(pos) * freq;

  if (i < half_dim) {
    pos_embeddings[idx] = sin(angle);
  } else {
    pos_embeddings[idx] = cos(angle);
  }
}
