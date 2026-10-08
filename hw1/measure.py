"""
Измерения на GPU: latency, peak memory, energy, OOM.

Все измерения — под torch.inference_mode(), eval(), FP32.
"""

import csv
import time
import torch
import numpy as np

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


# ---------- флаги (задаются один раз в main.py) ----------
def configure_backends():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


# ---------- latency ----------
@torch.inference_mode()
def measure_latency(model, x, n_runs: int = 50) -> float:
    """Median latency in seconds per forward pass."""
    # warm-up
    for _ in range(3):
        model(x)

    if x.is_cuda:
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
        for i in range(n_runs):
            starts[i].record()
            model(x)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]  # ms
        return float(np.median(times)) / 1000.0
    else:
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
        return float(np.median(times))


# ---------- peak memory ----------
@torch.inference_mode()
def measure_peak_memory(model, x):
    """Peak allocated memory in bytes during one forward pass."""
    if not x.is_cuda:
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = model(x)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated())


# ---------- energy ----------
@torch.inference_mode()
def measure_energy(model, x, handle, n_runs: int = 50) -> float:
    """Median energy per forward pass in Joules (whole GPU)."""
    if handle is None:
        return float("nan")
    # warm-up
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()

    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
    for _ in range(n_runs):
        model(x)
    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    return (e1 - e0) / 1000.0 / n_runs  # J per run


# ---------- полный прогон по гриду ----------
def run_grid(model, sizes, batches, n_runs=50, handle=None, is_validation=False):
    """
    Возвращает список словарей с измерениями для всех (S, B).
    Ловит CUDA OOM.
    """
    device = next(model.parameters()).device
    rows = []

    for S in sizes:
        for B in batches:
            row = {"S": int(S), "B": int(B), "is_validation": bool(is_validation)}
            try:
                x = torch.randn(B, 3, S, S, device=device)

                row["latency_s"]  = measure_latency(model, x, n_runs=n_runs)
                row["memory_b"]   = measure_peak_memory(model, x)
                row["energy_j"]   = measure_energy(model, x, handle, n_runs=n_runs)
                row["oom"]        = False

                del x
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                row["latency_s"] = float("nan")
                row["memory_b"]  = float("nan")
                row["energy_j"]  = float("nan")
                row["oom"]       = True
                torch.cuda.empty_cache()

            rows.append(row)
            print(f"[{'val' if is_validation else 'base'}] "
                  f"S={S:4d} B={B:4d}  "
                  f"lat={row['latency_s']*1e3:8.3f} ms  "
                  f"mem={row['memory_b']/2**20:7.1f} MiB  "
                  f"E={row['energy_j']:.4f} J  "
                  f"{'OOM' if row['oom'] else ''}")

    return rows


def save_csv(rows, path):
    keys = ["S", "B", "latency_s", "memory_b", "energy_j", "oom", "is_validation"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})