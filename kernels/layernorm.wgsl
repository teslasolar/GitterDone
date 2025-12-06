// Layer Normalization Kernel
// LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

struct Params {
  size: u32,        // Number of elements per normalization group
  num_groups: u32,  // Number of groups (e.g., batch_size * seq_len)
  epsilon: f32,     // Small constant for numerical stability (usually 1e-5)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;     // Scale parameter
@group(0) @binding(2) var<storage, read> beta: array<f32>;      // Shift parameter
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for reduction
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sum_sq: array<f32, 256>;
var<workgroup> group_mean: f32;
var<workgroup> group_var: f32;

@compute @workgroup_size(256)
fn layernorm_forward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let group_idx = wg_id.x;
  let local_idx = local_id.x;
  let base_offset = group_idx * params.size;

  if (group_idx >= params.num_groups) {
    return;
  }

  // Step 1: Compute partial sums for mean
  var local_sum = 0.0;
  var local_sum_sq = 0.0;

  let elements_per_thread = (params.size + 255u) / 256u;
  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx];
      local_sum += val;
      local_sum_sq += val * val;
    }
  }

  shared_sum[local_idx] = local_sum;
  shared_sum_sq[local_idx] = local_sum_sq;

  workgroupBarrier();

  // Step 2: Parallel reduction for sum
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local_idx < stride) {
      shared_sum[local_idx] += shared_sum[local_idx + stride];
      shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
    }
    workgroupBarrier();
  }

  // Step 3: Compute mean and variance
  if (local_idx == 0u) {
    let mean = shared_sum[0] / f32(params.size);
    let mean_sq = shared_sum_sq[0] / f32(params.size);
    group_mean = mean;
    group_var = mean_sq - mean * mean;
  }

  workgroupBarrier();

  // Step 4: Normalize and apply affine transformation
  let inv_std = 1.0 / sqrt(group_var + params.epsilon);

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx];
      let normalized = (val - group_mean) * inv_std;
      output[base_offset + elem_idx] = gamma[elem_idx] * normalized + beta[elem_idx];
    }
  }
}

// RMSNorm variant: RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon)
@compute @workgroup_size(256)
fn rmsnorm_forward(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let group_idx = wg_id.x;
  let local_idx = local_id.x;
  let base_offset = group_idx * params.size;

  if (group_idx >= params.num_groups) {
    return;
  }

  // Compute sum of squares
  var local_sum_sq = 0.0;
  let elements_per_thread = (params.size + 255u) / 256u;

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx];
      local_sum_sq += val * val;
    }
  }

  shared_sum_sq[local_idx] = local_sum_sq;

  workgroupBarrier();

  // Parallel reduction
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local_idx < stride) {
      shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
    }
    workgroupBarrier();
  }

  // Compute RMS
  if (local_idx == 0u) {
    let mean_sq = shared_sum_sq[0] / f32(params.size);
    group_var = mean_sq;  // Reusing group_var for RMS
  }

  workgroupBarrier();

  let inv_rms = 1.0 / sqrt(group_var + params.epsilon);

  // Apply RMSNorm
  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx];
      output[base_offset + elem_idx] = gamma[elem_idx] * val * inv_rms;
    }
  }
}
