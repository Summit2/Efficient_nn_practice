"""
МНК-калибровка theta для latency и energy.

Модель:
    y_lat  = max(theta_lat[0] * t_compute, theta_lat[1] * t_memory)
    y_en   = max(theta_en[0]  * E_compute, theta_en[1]  * E_memory)

Т.к. max нелинейный — линеаризуем: подгоняем на «доминирующей» ветке,
либо (проще и устойчивее) подгоняем без max, а потом проверяем max-модель.
Здесь используем ЛИНЕЙНУЮ сумму с последующим max на инференсе — так
theta интерпретируемы как «эффективность» каждой ветки.
"""

import json
import numpy as np

from equations import (
    flops, bytes_moved,
    PEAK_FLOPS, BANDWIDTH,
    E_PER_FLOP, E_PER_BYTE,
)


def _design_matrix(rows):
    """Возвращает t_compute, t_memory, E_compute, E_memory для всех не-OOM строк."""
    t_comp, t_mem, E_comp, E_mem = [], [], [], []
    for r in rows:
        if r.get("oom", False):
            continue
        S, B = r["S"], r["B"]
        t_comp.append(flops(S, B) / PEAK_FLOPS)
        t_mem.append(bytes_moved(S, B) / BANDWIDTH)
        E_comp.append(flops(S, B) * E_PER_FLOP)
        E_mem.append(bytes_moved(S, B) * E_PER_BYTE)
    return (np.array(t_comp), np.array(t_mem),
            np.array(E_comp), np.array(E_mem))


def calibrate(rows):
    """
    Возвращает dict с theta_latency, theta_energy и R^2.
    Использует только не-OOM строки.
    """
    y_lat, y_en = [], []
    for r in rows:
        if r.get("oom", False):
            continue
        y_lat.append(r["latency_s"])
        y_en.append(r["energy_j"])
    y_lat = np.array(y_lat)
    y_en  = np.array(y_en)

    t_comp, t_mem, E_comp, E_mem = _design_matrix(rows)

    # ---- latency ----
    X_lat = np.column_stack([t_comp, t_mem])
    theta_lat, *_ = np.linalg.lstsq(X_lat, y_lat, rcond=None)

    # ---- energy ----
    X_en = np.column_stack([E_comp, E_mem])
    theta_en, *_ = np.linalg.lstsq(X_en, y_en, rcond=None)

    # ---- R^2 ----
    def r2(y, yp):
        ss_res = np.sum((y - yp) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot

    # для отчёта: линейная подгонка
    r2_lat_lin = r2(y_lat, X_lat @ theta_lat)
    r2_en_lin  = r2(y_en,  X_en  @ theta_en)

    # для отчёта: max-модель
    lat_max = np.maximum(theta_lat[0] * t_comp, theta_lat[1] * t_mem)
    en_max  = np.maximum(theta_en[0]  * E_comp, theta_en[1]  * E_mem)
    r2_lat_max = r2(y_lat, lat_max)
    r2_en_max  = r2(y_en,  en_max)

    return {
        "theta_latency": theta_lat.tolist(),
        "theta_energy":  theta_en.tolist(),
        "r2_latency_linear": float(r2_lat_lin),
        "r2_energy_linear":  float(r2_en_lin),
        "r2_latency_max":    float(r2_lat_max),
        "r2_energy_max":     float(r2_en_max),
    }


def save_theta(theta_dict, path):
    with open(path, "w") as f:
        json.dump(theta_dict, f, indent=2)


def load_theta(path):
    with open(path) as f:
        return json.load(f)