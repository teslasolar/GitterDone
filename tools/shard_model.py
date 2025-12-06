#!/usr/bin/env python3
"""
Model Sharding Tool for WebGPU Model Runner

Converts model weights to sharded binary format for Git LFS storage.
Supports ONNX, PyTorch, and raw binary formats.

Usage:
    python shard_model.py model.onnx --output ./shards --size 50MB
    python shard_model.py model.pt --format pytorch --output ./shards
    python shard_model.py weights.bin --format raw --output ./shards
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

def parse_size(size_str):
    """Parse size string like '50MB' to bytes."""
    size_str = size_str.strip().upper()
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-len(suffix)]) * mult)
    return int(size_str)

def shard_binary(data, shard_size, output_dir):
    """Split binary data into shards."""
    shards = []
    offset = 0
    shard_idx = 0

    while offset < len(data):
        chunk = data[offset:offset + shard_size]
        shard_name = f'shard_{shard_idx:03d}.bin'
        shard_path = output_dir / shard_name

        with open(shard_path, 'wb') as f:
            f.write(chunk)

        shards.append({
            'index': shard_idx,
            'name': shard_name,
            'size': len(chunk),
            'offset': offset
        })

        offset += shard_size
        shard_idx += 1
        print(f'  Created {shard_name} ({len(chunk)} bytes)')

    return shards

def load_onnx(model_path):
    """Load ONNX model and extract weights."""
    try:
        import onnx
        import numpy as np
    except ImportError:
        print("Error: onnx and numpy required. Install with: pip install onnx numpy")
        sys.exit(1)

    model = onnx.load(model_path)
    weights = {}

    for initializer in model.graph.initializer:
        tensor = onnx.numpy_helper.to_array(initializer)
        weights[initializer.name] = tensor.astype(np.float32)

    # Concatenate all weights into single buffer
    weight_list = list(weights.values())
    total_size = sum(w.nbytes for w in weight_list)

    buffer = bytearray(total_size)
    offset = 0
    layer_info = {}

    for name, weight in weights.items():
        flat = weight.flatten().tobytes()
        buffer[offset:offset + len(flat)] = flat
        layer_info[name] = {
            'offset': offset,
            'size': len(flat),
            'shape': list(weight.shape),
            'dtype': 'float32'
        }
        offset += len(flat)

    return bytes(buffer), layer_info, model

def load_pytorch(model_path):
    """Load PyTorch model and extract weights."""
    try:
        import torch
        import numpy as np
    except ImportError:
        print("Error: torch and numpy required. Install with: pip install torch numpy")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location='cpu')
    weights = {}

    for name, tensor in state_dict.items():
        weights[name] = tensor.numpy().astype(np.float32)

    # Build buffer
    total_size = sum(w.nbytes for w in weights.values())
    buffer = bytearray(total_size)
    offset = 0
    layer_info = {}

    for name, weight in weights.items():
        flat = weight.flatten().tobytes()
        buffer[offset:offset + len(flat)] = flat
        layer_info[name] = {
            'offset': offset,
            'size': len(flat),
            'shape': list(weight.shape),
            'dtype': 'float32'
        }
        offset += len(flat)

    return bytes(buffer), layer_info, None

def generate_demo_weights(config_path, output_dir):
    """Generate random weights based on config for demo purposes."""
    import random
    import struct

    with open(config_path) as f:
        config = json.load(f)

    # Calculate total weight size
    total_size = 0
    for layer in config.get('layers', []):
        if 'weight_size' in layer:
            total_size = max(total_size, layer.get('weight_offset', 0) + layer['weight_size'])
        if 'bias_size' in layer:
            total_size = max(total_size, layer.get('bias_offset', 0) + layer['bias_size'])

    # Add some padding
    total_size = max(total_size, config.get('shard_size', 51200))

    print(f"Generating {total_size} bytes of demo weights...")

    # Generate random float32 weights
    num_floats = total_size // 4
    random.seed(42)  # Reproducible

    buffer = bytearray()
    for i in range(num_floats):
        # Xavier-like initialization
        val = (random.random() - 0.5) * 0.1
        buffer.extend(struct.pack('f', val))

    # Pad to exact size
    while len(buffer) < total_size:
        buffer.append(0)

    # Write shard
    shard_path = output_dir / 'shard_000.bin'
    with open(shard_path, 'wb') as f:
        f.write(buffer)

    print(f"Created {shard_path} ({len(buffer)} bytes)")

    return [{
        'index': 0,
        'name': 'shard_000.bin',
        'size': len(buffer),
        'offset': 0
    }]

def main():
    parser = argparse.ArgumentParser(description='Shard model weights for WebGPU inference')
    parser.add_argument('input', help='Input model file or config.json for demo generation')
    parser.add_argument('--output', '-o', default='./shards', help='Output directory')
    parser.add_argument('--size', '-s', default='50MB', help='Shard size (e.g., 50MB)')
    parser.add_argument('--format', '-f', choices=['onnx', 'pytorch', 'raw', 'demo'],
                        default='auto', help='Input format')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    shard_size = parse_size(args.size)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect format
    fmt = args.format
    if fmt == 'auto':
        suffix = input_path.suffix.lower()
        if suffix == '.onnx':
            fmt = 'onnx'
        elif suffix in ['.pt', '.pth', '.bin']:
            fmt = 'pytorch'
        elif suffix == '.json':
            fmt = 'demo'
        else:
            fmt = 'raw'

    print(f"Processing {input_path} (format: {fmt})")
    print(f"Output: {output_dir}")
    print(f"Shard size: {shard_size} bytes")
    print()

    layer_info = {}

    if fmt == 'demo':
        shards = generate_demo_weights(input_path, output_dir)
    elif fmt == 'onnx':
        data, layer_info, _ = load_onnx(input_path)
        shards = shard_binary(data, shard_size, output_dir)
    elif fmt == 'pytorch':
        data, layer_info, _ = load_pytorch(input_path)
        shards = shard_binary(data, shard_size, output_dir)
    else:
        # Raw binary
        with open(input_path, 'rb') as f:
            data = f.read()
        shards = shard_binary(data, shard_size, output_dir)

    # Write manifest
    manifest = {
        'shards': len(shards),
        'shard_size': shard_size,
        'total_size': sum(s['size'] for s in shards),
        'shard_list': shards,
        'layers': layer_info
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Created {len(shards)} shards")
    print(f"Manifest: {manifest_path}")
    print("Done!")

if __name__ == '__main__':
    main()
