from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROCESSED = RESULTS / "processed"
FIGURES = ROOT / "paper" / "figures"

PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

# DirectML adapter enumeration:
# 0 = Intel UHD Graphics
# 1 = NVIDIA GeForce RTX 3050 6GB Laptop GPU
DEVICE_NAMES = {
    0: "Intel UHD",
    1: "RTX 3050",
}

def model_label(path):
    name = str(path).lower()
    if "squeeze" in name:
        return "SqueezeNet 1.1"
    if "mobilenet" in name:
        return "MobileNetV2"
    return Path(name).name

def device_label(device):
    device = int(device)
    return DEVICE_NAMES.get(device, f"Unknown DML device {device}")

def median_iqr(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan
    return (
        float(values.median()),
        float(values.quantile(0.75) - values.quantile(0.25)),
    )

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path)

# ---------------------------------------------------------------------
# Load the three datasets independently. Never combine their rows.
# ---------------------------------------------------------------------
cold = require_file(RESULTS / "coldstart_raw.csv")
old_hybrid = require_file(RESULTS / "hybrid_raw.csv")
dispatcher = require_file(RESULTS / "hybrid_crossover_raw.csv")

cold["model_name"] = cold["model"].apply(model_label)
cold["backend"] = cold.apply(
    lambda r: f"CPU ({int(r['param'])}T)"
    if r["ep"] == "cpu"
    else f"{device_label(r['param'])} (DML-{int(r['param'])})",
    axis=1,
)

old_hybrid["model_name"] = old_hybrid["model"].apply(model_label)
old_hybrid["gpu_name"] = old_hybrid["gpu_device"].apply(device_label)

dispatcher["model_name"] = dispatcher["model"].apply(model_label)
dispatcher["gpu_name"] = dispatcher["gpu_device"].apply(device_label)

# ---------------------------------------------------------------------
# 1. Baseline cold-start summary
# ---------------------------------------------------------------------
cold_rows = []

for (model, backend), group in cold.groupby(["model_name", "backend"]):
    create_med, create_iqr = median_iqr(group["session_create_ms"])
    first_med, first_iqr = median_iqr(group["first_inf_ms"])
    steady_med, steady_iqr = median_iqr(group["steady_median_ms"])
    p95_med, _ = median_iqr(group["steady_p95_ms"])
    gap_med, gap_iqr = median_iqr(group["cold_gap_ms"])

    cold_rows.append({
        "model": model,
        "backend": backend,
        "n_trials": len(group),
        "create_median_ms": create_med,
        "create_iqr_ms": create_iqr,
        "first_inference_median_ms": first_med,
        "first_inference_iqr_ms": first_iqr,
        "cold_total_ms": create_med + first_med,
        "steady_median_ms": steady_med,
        "steady_median_iqr_ms": steady_iqr,
        "steady_p95_median_ms": p95_med,
        "cold_gap_median_ms": gap_med,
        "cold_gap_iqr_ms": gap_iqr,
    })

summary = pd.DataFrame(cold_rows)
summary.to_csv(PROCESSED / "coldstart_summary.csv", index=False)

# ---------------------------------------------------------------------
# 2. Historical hybrid summary (kept separate from the new dispatcher)
# ---------------------------------------------------------------------
old_rows = []

for (model, gpu), group in old_hybrid.groupby(["model_name", "gpu_name"]):
    ttfi_med, ttfi_iqr = median_iqr(group["ttfi_ms"])
    switch_med, _ = median_iqr(group["switch_frame"])
    init_med, _ = median_iqr(group["bg_gpu_init_ms"])
    cpu_med, _ = median_iqr(group["cpu_phase_med_ms"])
    gpu_med, _ = median_iqr(group["gpu_phase_med_ms"])
    wall_med, _ = median_iqr(group["total_wall_ms"])

    old_rows.append({
        "model": model,
        "gpu": gpu,
        "n_trials": len(group),
        "ttfi_median_ms": ttfi_med,
        "ttfi_iqr_ms": ttfi_iqr,
        "switch_frame_median": switch_med,
        "gpu_init_median_ms": init_med,
        "cpu_phase_median_ms": cpu_med,
        "gpu_phase_median_ms": gpu_med,
        "total_wall_median_ms": wall_med,
    })

old_summary = pd.DataFrame(old_rows)
old_summary.to_csv(PROCESSED / "old_hybrid_summary.csv", index=False)

# ---------------------------------------------------------------------
# 3. Corrected dispatcher summary (kept separate)
# ---------------------------------------------------------------------
dispatcher_rows = []

for (model, gpu, decision), group in dispatcher.groupby(
    ["model_name", "gpu_name", "decision"]
):
    ttfi_med, ttfi_iqr = median_iqr(group["ttfi_ms"])
    switch_med, _ = median_iqr(group["switch_frame"])
    init_med, _ = median_iqr(group["bg_gpu_init_ms"])
    cpu_med, _ = median_iqr(group["cpu_phase_med_ms"])
    gpu_med, _ = median_iqr(group["gpu_phase_med_ms"])
    workload_med, _ = median_iqr(group["workload_wall_ms"])
    process_med, _ = median_iqr(group["process_wall_ms"])

    dispatcher_rows.append({
        "model": model,
        "gpu": gpu,
        "decision": decision,
        "n_trials": len(group),
        "ttfi_median_ms": ttfi_med,
        "ttfi_iqr_ms": ttfi_iqr,
        "switch_frame_median": switch_med,
        "gpu_init_median_ms": init_med,
        "cpu_phase_median_ms": cpu_med,
        "gpu_phase_median_ms": gpu_med,
        "workload_wall_median_ms": workload_med,
        "process_wall_median_ms": process_med,
    })

dispatcher_summary = pd.DataFrame(dispatcher_rows)
dispatcher_summary.to_csv(
    PROCESSED / "dispatcher_summary.csv", index=False
)

# ---------------------------------------------------------------------
# 4. Strict break-even calculation against CPU (4 threads)
#
# GPU wins when:
# G_cold + (N - 1) * G_step < C_cold + (N - 1) * C_step
#
# If G_step < C_step, the smallest integer N satisfying the strict
# inequality is floor((G_cold - C_cold)/(C_step - G_step)) + 2.
# ---------------------------------------------------------------------
print("\n=== BASELINE COLD-START SUMMARY ===")
print(summary.round(4).to_string(index=False))

print("\n=== HISTORICAL HYBRID SUMMARY (SEPARATE DATASET) ===")
print(old_summary.round(4).to_string(index=False))

print("\n=== CORRECTED DISPATCHER SUMMARY (SEPARATE DATASET) ===")
print(dispatcher_summary.round(4).to_string(index=False))

print("\n=== STRICT BREAK-EVEN ANALYSIS VS CPU (4T) ===")

crossover_rows = []

for model in ["SqueezeNet 1.1", "MobileNetV2"]:
    sub = summary[summary["model"] == model].set_index("backend")
    cpu_key = "CPU (4T)"

    if cpu_key not in sub.index:
        raise ValueError(f"Missing {cpu_key} baseline for {model}")

    cpu_cold = float(sub.loc[cpu_key, "cold_total_ms"])
    cpu_step = float(sub.loc[cpu_key, "steady_median_ms"])

    for device in [0, 1]:
        gpu_name = device_label(device)
        gpu_key = f"{gpu_name} (DML-{device})"

        if gpu_key not in sub.index:
            raise ValueError(f"Missing {gpu_key} baseline for {model}")

        gpu_cold = float(sub.loc[gpu_key, "cold_total_ms"])
        gpu_step = float(sub.loc[gpu_key, "steady_median_ms"])

        if gpu_step >= cpu_step:
            frames_to_win = None
            result = "No crossover: GPU steady-state is not faster"
        else:
            ratio = (gpu_cold - cpu_cold) / (cpu_step - gpu_step)
            frames_to_win = max(1, math.floor(ratio) + 2)
            result = f"{frames_to_win} frames"

        crossover_rows.append({
            "model": model,
            "gpu": gpu_name,
            "cpu_cold_ms": cpu_cold,
            "gpu_cold_ms": gpu_cold,
            "cpu_steady_ms": cpu_step,
            "gpu_steady_ms": gpu_step,
            "break_even_frames": frames_to_win,
            "result": result,
        })

        print(
            f"{model} | {gpu_name}: "
            f"CPU cold={cpu_cold:.3f} ms, GPU cold={gpu_cold:.3f} ms, "
            f"CPU steady={cpu_step:.4f} ms, GPU steady={gpu_step:.4f} ms "
            f"=> {result}"
        )

crossover = pd.DataFrame(crossover_rows)
crossover.to_csv(PROCESSED / "crossover_summary.csv", index=False)

# ---------------------------------------------------------------------
# 5. Figure 1: Baseline cold-start versus steady-state
# ---------------------------------------------------------------------
backend_order = [
    "CPU (1T)",
    "CPU (4T)",
    "Intel UHD (DML-0)",
    "RTX 3050 (DML-1)",
]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

for ax, model in zip(axes, ["SqueezeNet 1.1", "MobileNetV2"]):
    sub = (
        summary[summary["model"] == model]
        .set_index("backend")
        .reindex(backend_order)
    )

    x = np.arange(len(backend_order))
    cold_bars = ax.bar(
        x - 0.18, sub["cold_total_ms"], width=0.34,
        label="Cold total (init + first inference)"
    )
    steady_bars = ax.bar(
        x + 0.18, sub["steady_median_ms"], width=0.34,
        label="Steady-state median"
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["CPU\n1T", "CPU\n4T", "Intel\niGPU", "RTX\n3050"])
    ax.set_title(model)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bars, fmt in [(cold_bars, "{:.1f}"), (steady_bars, "{:.2f}")]:
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.10,
                    fmt.format(height),
                    ha="center", va="bottom", fontsize=7,
                )

axes[0].set_ylabel("Latency (ms, logarithmic scale)")
axes[1].legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "fig1_coldstart_breakdown.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig1_coldstart_breakdown.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 6. Figure 2: Baseline cumulative latency model only.
#    It is a model derived from measured medians, not a dispatcher trace.
# ---------------------------------------------------------------------
frames = np.arange(1, 1001)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

line_specs = [
    ("CPU (4T)", "--"),
    ("Intel UHD (DML-0)", "-."),
    ("RTX 3050 (DML-1)", ":"),
]

for ax, model in zip(axes, ["SqueezeNet 1.1", "MobileNetV2"]):
    sub = summary[summary["model"] == model].set_index("backend")

    for backend, linestyle in line_specs:
        cold_ms = float(sub.loc[backend, "cold_total_ms"])
        step_ms = float(sub.loc[backend, "steady_median_ms"])
        cumulative_ms = cold_ms + (frames - 1) * step_ms

        ax.plot(
            frames, cumulative_ms / 1000.0,
            linestyle=linestyle, linewidth=1.8,
            label=backend,
        )

    model_cross = crossover[
        (crossover["model"] == model)
        & crossover["break_even_frames"].notna()
    ]

    for _, row in model_cross.iterrows():
        n = int(row["break_even_frames"])
        if n <= 1000:
            ax.axvline(n, color="grey", linestyle=":", linewidth=1)
            ax.annotate(
                f"{row['gpu']} break-even: {n} frames",
                xy=(n, 0.92), xycoords=("data", "axes fraction"),
                xytext=(5, -3), textcoords="offset points",
                fontsize=7, rotation=90, va="top",
            )

    ax.set_title(model)
    ax.set_xlabel("Completed inference frames")
    ax.grid(True, linestyle="--", alpha=0.35)

axes[0].set_ylabel("Cumulative modeled time (seconds)")
axes[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "fig2_cumulative_crossover.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig2_cumulative_crossover.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 7. Figure 3: Corrected dispatcher diagnostics.
#    Panel A: observed TTFI, annotated with actual decision.
#    Panel B: CPU/GPU phase medians for configurations with GPU timings.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

plot_data = dispatcher_summary.copy()
plot_data["configuration"] = (
    plot_data["model"].str.replace(" 1.1", "", regex=False)
    + "\n" + plot_data["gpu"]
)

x = np.arange(len(plot_data))
bars = axes[0].bar(x, plot_data["ttfi_median_ms"])
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot_data["configuration"], rotation=0)
axes[0].set_ylabel("TTFI (ms)")
axes[0].set_title("Corrected dispatcher: observed TTFI")
axes[0].grid(axis="y", linestyle="--", alpha=0.35)

for bar, (_, row) in zip(bars, plot_data.iterrows()):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.03,
        str(row["decision"]).replace("_", "\n"),
        ha="center", va="bottom", fontsize=6.5,
    )

selected = dispatcher_summary[
    dispatcher_summary["gpu_phase_median_ms"].notna()
].copy()

if not selected.empty:
    labels = selected["model"].str.replace(" 1.1", "", regex=False) + "\n" + selected["gpu"]
    xx = np.arange(len(selected))
    axes[1].bar(
        xx - 0.18, selected["cpu_phase_median_ms"],
        width=0.34, label="CPU phase",
    )
    axes[1].bar(
        xx + 0.18, selected["gpu_phase_median_ms"],
        width=0.34, label="GPU phase",
    )
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
else:
    axes[1].text(
        0.5, 0.5, "No GPU phase timings recorded",
        ha="center", va="center", transform=axes[1].transAxes,
    )

axes[1].set_ylabel("Per-frame phase median (ms)")
axes[1].set_title("Configurations with GPU execution")
axes[1].grid(axis="y", linestyle="--", alpha=0.35)

fig.tight_layout()
fig.savefig(FIGURES / "fig3_hybrid_ttfi_comparison.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig3_hybrid_ttfi_comparison.png", bbox_inches="tight")
plt.close(fig)

print("\nSaved processed summaries to:", PROCESSED)
print("Regenerated figures in:", FIGURES)
print("\nAnalysis completed. Raw CSV datasets were not modified.")
