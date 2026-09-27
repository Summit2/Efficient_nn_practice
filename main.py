"""
Homework 1 — точка входа.

Запуск:
    python main.py

Что делает:
    1. Настраивает backend-флаги.
    2. Строит модель на GPU.
    3. Прогоняет base grid (11 S × 9 B) + validation grid (4 S × 3 B).
    4. Калибрует theta по base grid.
    5. Считает предсказания на validation grid и строит графики.
    6. Сохраняет measurements.csv, theta.json, figures/*.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models import build_model
from measure import configure_backends, run_grid, save_csv
from equations import flops, memory, latency, energy, bytes_moved
from calibrate import calibrate, save_theta

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


# ---------------- grids ----------------
BASE_SIZES   = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# validation: 4 S из [32,512] не из base + 3 B не степени двойки
VAL_SIZES   = [48, 96, 160, 320]
VAL_BATCHES = [3, 12, 96]


def ensure_dirs():
    os.makedirs("results/figures", exist_ok=True)


# ---------------- plots ----------------
def plot_latency(rows_base, rows_val, theta_lat, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for B in sorted({r["B"] for r in rows_base}):
        pts = sorted([r for r in rows_base if r["B"] == B and not r["oom"]],
                     key=lambda r: r["S"])
        if not pts:
            continue
        S = np.array([p["S"] for p in pts])
        y = np.array([p["latency_s"] for p in pts]) * 1e3
        ax.plot(S, y, "o", ms=4, alpha=0.6, label=f"measured B={B}")
        S_curve = np.linspace(32, 512, 60)
        y_curve = latency(S_curve, B, theta_lat) * 1e3
        ax.plot(S_curve, y_curve, "-", lw=1.0, alpha=0.7)

    # validation
    for r in rows_val:
        if r["oom"]:
            continue
        y_pred = latency(r["S"], r["B"], theta_lat) * 1e3
        ax.scatter([r["S"]], [r["latency_s"] * 1e3],
                   marker="*", s=80, facecolors="none",
                   edgecolors="k", zorder=5)
        ax.scatter([r["S"]], [y_pred],
                   marker="x", s=40, color="k", zorder=5)

    ax.set_xlabel("Image size S")
    ax.set_ylabel("Latency, ms")
    ax.set_title("Latency: measured (points) vs model (lines)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_energy(rows_base, rows_val, theta_en, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for B in sorted({r["B"] for r in rows_base}):
        pts = sorted([r for r in rows_base if r["B"] == B and not r["oom"]],
                     key=lambda r: r["S"])
        if not pts:
            continue
        S = np.array([p["S"] for p in pts])
        y = np.array([p["energy_j"] for p in pts])
        ax.plot(S, y, "o", ms=4, alpha=0.6, label=f"measured B={B}")
        S_curve = np.linspace(32, 512, 60)
        y_curve = energy(S_curve, B, theta_en)
        ax.plot(S_curve, y_curve, "-", lw=1.0, alpha=0.7)

    for r in rows_val:
        if r["oom"]:
            continue
        y_pred = energy(r["S"], r["B"], theta_en)
        ax.scatter([r["S"]], [r["energy_j"]],
                   marker="*", s=80, facecolors="none",
                   edgecolors="k", zorder=5)
        ax.scatter([r["S"]], [y_pred],
                   marker="x", s=40, color="k", zorder=5)

    ax.set_xlabel("Image size S")
    ax.set_ylabel("Energy, J")
    ax.set_title("Energy: measured (points) vs model (lines)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_memory(rows_base, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for B in sorted({r["B"] for r in rows_base}):
        pts = sorted([r for r in rows_base if r["B"] == B],
                     key=lambda r: r["S"])
        S_meas = [p["S"] for p in pts if not p["oom"]]
        y_meas = [p["memory_b"] / 2**20 for p in pts if not p["oom"]]
        S_oom  = [p["S"] for p in pts if p["oom"]]
        ax.plot(S_meas, y_meas, "o-", ms=4, alpha=0.7, label=f"measured B={B}")
        for s in S_oom:
            ax.axvline(s, color="r", alpha=0.05)

        S_curve = np.linspace(32, 512, 60)
        y_curve = memory(S_curve, B) / 2**20
        ax.plot(S_curve, y_curve, "--", lw=1.0, alpha=0.5)

    ax.set_xlabel("Image size S")
    ax.set_ylabel("Peak memory, MiB")
    ax.set_title("Memory: measured (solid) vs model (dashed), red = OOM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------- main ----------------
def main():
    ensure_dirs()
    configure_backends()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")

    model = build_model(device=device)

    handle = None
    if device == "cuda" and _HAS_NVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("NVML: OK (энергия будет измеряться)")
        except Exception as e:
            print(f"NVML init failed: {e}")
    else:
        print("NVML недоступен — энергия будет NaN")

    N_RUNS = 50

    # ---------- 1) base grid ----------
    print("\n=== BASE GRID ===")
    rows_base = run_grid(model, BASE_SIZES, BASE_BATCHES,
                         n_runs=N_RUNS, handle=handle, is_validation=False)

    # ---------- 2) калибровка ----------
    print("\n=== CALIBRATION ===")
    theta = calibrate(rows_base)
    print(f"theta_latency = {theta['theta_latency']}")
    print(f"theta_energy  = {theta['theta_energy']}")
    print(f"R^2 latency (linear) = {theta['r2_latency_linear']:.4f}")
    print(f"R^2 latency (max)    = {theta['r2_latency_max']:.4f}")
    print(f"R^2 energy  (linear) = {theta['r2_energy_linear']:.4f}")
    print(f"R^2 energy  (max)    = {theta['r2_energy_max']:.4f}")
    save_theta(theta, "results/theta.json")

    theta_lat = tuple(theta["theta_latency"])
    theta_en  = tuple(theta["theta_energy"])

    # ---------- 3) validation grid ----------
    print("\n=== VALIDATION GRID ===")
    rows_val = run_grid(model, VAL_SIZES, VAL_BATCHES,
                        n_runs=N_RUNS, handle=handle, is_validation=True)

    # ---------- 4) сохранить CSV ----------
    save_csv(rows_base + rows_val, "results/measurements.csv")
    print("\nSaved results/measurements.csv")

    # ---------- 5) графики ----------
    plot_latency(rows_base, rows_val, theta_lat,
                 "results/figures/latency.png")
    plot_energy(rows_base, rows_val, theta_en,
                "results/figures/energy.png")
    plot_memory(rows_base, "results/figures/memory.png")
    print("Saved results/figures/*.png")


if __name__ == "__main__":
    main()