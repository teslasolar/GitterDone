// Softmax Kernel
// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

struct Params {
  size: u32,          // Number of elements per softmax group
  num_groups: u32,    // Number of groups (e.g., batch_size)
  temperature: f32,   // Temperature for scaling (default 1.0)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for reduction
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> group_max: f32;
var<workgroup> group_sum: f32;

@compute @workgroup_size(256)
fn softmax_forward(
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

  // Step 1: Find maximum value (for numerical stability)
  var local_max = -1e38;
  let elements_per_thread = (params.size + 255u) / 256u;

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx] / params.temperature;
      local_max = max(local_max, val);
    }
  }

  shared_max[local_idx] = local_max;

  workgroupBarrier();

  // Parallel reduction for max
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local_idx < stride) {
      shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
    }
    workgroupBarrier();
  }

  if (local_idx == 0u) {
    group_max = shared_max[0];
  }

  workgroupBarrier();

  // Step 2: Compute exp(x - max) and sum
  var local_sum = 0.0;

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx] / params.temperature;
      let exp_val = exp(val - group_max);
      output[base_offset + elem_idx] = exp_val;
      local_sum += exp_val;
    }
  }

  shared_sum[local_idx] = local_sum;

  workgroupBarrier();

  // Parallel reduction for sum
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local_idx < stride) {
      shared_sum[local_idx] += shared_sum[local_idx + stride];
    }
    workgroupBarrier();
  }

  if (local_idx == 0u) {
    group_sum = shared_sum[0];
  }

  workgroupBarrier();

  // Step 3: Normalize
  let inv_sum = 1.0 / group_sum;

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      output[base_offset + elem_idx] *= inv_sum;
    }
  }
}

// Log softmax (more numerically stable for cross-entropy loss)
@compute @workgroup_size(256)
fn log_softmax_forward(
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

  // Find max
  var local_max = -1e38;
  let elements_per_thread = (params.size + 255u) / 256u;

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx] / params.temperature;
      local_max = max(local_max, val);
    }
  }

  shared_max[local_idx] = local_max;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local_idx < stride) {
      shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
    }
    workgroupBarrier();
  }

  if (local_idx == 0u) {
    group_max = shared_max[0];
  }
  workgroupBarrier();

  // Compute sum of exp
  var local_sum = 0.0;

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx] / params.temperature;
      local_sum += exp(val - group_max);
    }
  }

  shared_sum[local_idx] = local_sum;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local_idx < stride) {
      shared_sum[local_idx] += shared_sum[local_idx + stride];
    }
    workgroupBarrier();
  }

  if (local_idx == 0u) {
    group_sum = shared_sum[0];
  }
  workgroupBarrier();

  // Compute log softmax: x - max - log(sum(exp(x - max)))
  let log_sum = log(group_sum);

  for (var i = 0u; i < elements_per_thread; i++) {
    let elem_idx = local_idx * elements_per_thread + i;
    if (elem_idx < params.size) {
      let val = input[base_offset + elem_idx] / params.temperature;
      output[base_offset + elem_idx] = val - group_max - log_sum;
    }
  }
}

// Top-K sampling helper (marks top-k indices)
struct TopKParams {
  size: u32,
  k: u32,
}

@group(0) @binding(3) var<uniform> topk_params: TopKParams;
@group(0) @binding(4) var<storage, read_write> mask: array<u32>;

@compute @workgroup_size(256)
fn topk_mask(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  // This is a simplified version - full top-k requires sorting
  // For production, use a proper parallel sorting algorithm
  let idx = global_id.x;

  if (idx >= topk_params.size) {
    return;
  }

  // Count how many elements are larger than this one
  var count = 0u;
  let val = output[idx];

  for (var i = 0u; i < topk_params.size; i++) {
    if (output[i] > val) {
      count++;
    }
  }

  // Mark as 1 if in top-k, 0 otherwise
  mask[idx] = select(0u, 1u, count < topk_params.k);
}
