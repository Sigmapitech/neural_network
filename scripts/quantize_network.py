#!/usr/bin/env python3
import json
import sys

if len(sys.argv) < 3:
    print("Usage: quantize_model.py input.json output.json [quantize_level]")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
quantize_level = int(sys.argv[3]) if len(sys.argv) >= 4 else None

with open(input_file) as f:
    model = json.load(f)

quantized_weights = []
scales = []

for layer in model["parameters"]["weights"]:
    w_q = [
        [round(float(val), quantize_level or 1) for val in sub]
        for sub in layer
    ]

    quantized_weights.append(w_q)

model["parameters"]["weights"] = quantized_weights
model["parameters"]["quantization"] = {
    "type": "int8" if quantize_level is None else "rounding",
    "scheme": "symmetric" if quantize_level is None else "decimal",
    "per": "layer",
    "quantize_level": quantize_level,
}

with open(output_file, "w") as f:
    json.dump(model, f, indent=2)

print("Quantization complete.")
