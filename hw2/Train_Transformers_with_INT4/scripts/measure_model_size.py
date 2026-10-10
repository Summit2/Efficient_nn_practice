#!/usr/bin/env python3
"""Estimate model weight storage for FP16 and packed INT4.

This measures parameter storage, not the complete runtime footprint.  The
repository uses fake/simulated quantization during training, so an actual
checkpoint must be saved separately to measure its serialized file size.
"""

import argparse
import math


def human(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=int, default=109_483_778)
    parser.add_argument("--fp16-bytes", type=float, default=2.0)
    parser.add_argument("--int4-bits", type=int, default=4)
    parser.add_argument(
        "--scale-bytes",
        type=float,
        default=0.0,
        help="Optional extra bytes per quantization group (not included by default).",
    )
    parser.add_argument("--group-size", type=int, default=0)
    args = parser.parse_args()

    fp16 = args.parameters * args.fp16_bytes
    int4 = args.parameters * args.int4_bits / 8
    scales = 0.0
    if args.scale_bytes and args.group_size:
        scales = math.ceil(args.parameters / args.group_size) * args.scale_bytes
        int4 += scales

    print(f"parameters: {args.parameters:,}")
    print(f"fp16_weights_bytes: {fp16:.0f}")
    print(f"fp16_weights: {human(fp16)}")
    print(f"int4_packed_bytes: {int4:.0f}")
    print(f"int4_packed: {human(int4)}")
    if scales:
        print(f"scale_overhead: {human(scales)}")
    print(f"int4_vs_fp16_ratio: {int4 / fp16:.4f}")
    print(f"theoretical_saving: {100 * (1 - int4 / fp16):.2f}%")


if __name__ == "__main__":
    main()
