// Matrix Multiplication Kernel
// C = A @ B where A is MxK, B is KxN, C is MxN

struct Uniforms {
  M: u32,  // rows of A and C
  N: u32,  // cols of B and C
  K: u32,  // cols of A, rows of B
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let row = global_id.x;
  let col = global_id.y;
  let local_row = local_id.x;
  let local_col = local_id.y;

  var sum = 0.0;

  // Tiled matrix multiplication
  let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t = 0u; t < num_tiles; t++) {
    // Load tile of A into shared memory
    let a_row = row;
    let a_col = t * TILE_SIZE + local_col;
    if (a_row < uniforms.M && a_col < uniforms.K) {
      tile_a[local_row * TILE_SIZE + local_col] = a[a_row * uniforms.K + a_col];
    } else {
      tile_a[local_row * TILE_SIZE + local_col] = 0.0;
    }

    // Load tile of B into shared memory
    let b_row = t * TILE_SIZE + local_row;
    let b_col = col;
    if (b_row < uniforms.K && b_col < uniforms.N) {
      tile_b[local_row * TILE_SIZE + local_col] = b[b_row * uniforms.N + b_col];
    } else {
      tile_b[local_row * TILE_SIZE + local_col] = 0.0;
    }

    workgroupBarrier();

    // Compute partial dot product
    for (var k = 0u; k < TILE_SIZE; k++) {
      sum += tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
    }

    workgroupBarrier();
  }

  // Write result
  if (row < uniforms.M && col < uniforms.N) {
    c[row * uniforms.N + col] = sum;
  }
}
