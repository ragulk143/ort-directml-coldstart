import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

cold = pd.read_csv("results/coldstart_raw.csv")
hyb = pd.read_csv("results/hybrid_raw.csv")

def label_backend(row):
    if row["ep"] == "cpu":
        return f"CPU ({int(row['param'])}T)"
    return "RTX 3050 (DML-0)" if int(row["param"]) == 0 else "Intel UHD (DML-1)"

cold["backend"] = cold.apply(label_backend, axis=1)
cold["model_name"] = cold["model"].apply(lambda s: "SqueezeNet 1.1" if "squeeze" in s else "MobileNetV2")
hyb["model_name"] = hyb["model"].apply(lambda s: "SqueezeNet 1.1" if "squeeze" in s else "MobileNetV2")
hyb["gpu_name"] = hyb["gpu_device"].apply(lambda d: "RTX 3050 (DML-0)" if int(d) == 0 else "Intel UHD (DML-1)")

# 1. Summary statistics (Median, IQR, p95)
summary = cold.groupby(["model_name", "backend"]).agg(
    n_trials=("trial", "count"),
    create_med=("session_create_ms", "median"),
    create_iqr=("session_create_ms", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    first_med=("first_inf_ms", "median"),
    steady_med=("steady_median_ms", "median"),
    steady_p95=("steady_p95_ms", "median"),
    gap_med=("cold_gap_ms", "median"),
    gap_iqr=("cold_gap_ms", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
).reset_index()

summary["cold_total_ms"] = summary["create_med"] + summary["first_med"]
summary.to_csv("results/processed/coldstart_summary.csv", index=False)

hyb_summary = hyb.groupby(["model_name", "gpu_name"]).agg(
    n_trials=("trial", "count"),
    ttfi_med=("ttfi_ms", "median"),
    ttfi_iqr=("ttfi_ms", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    switch_frame_med=("switch_frame", "median"),
    bg_init_med=("bg_gpu_init_ms", "median"),
    cpu_phase_med=("cpu_phase_med_ms", "median"),
    gpu_phase_med=("gpu_phase_med_ms", "median"),
    p95_med=("overall_p95_ms", "median"),
    wall_800_med=("total_wall_ms", "median"),
).reset_index()
hyb_summary.to_csv("results/processed/hybrid_summary.csv", index=False)

print("=== TABLE 1: COLD-START vs STEADY-STATE SUMMARY ===")
print(summary.to_string(index=False))
print("\n=== TABLE 2: HYBRID DISPATCHER SUMMARY (800 FRAMES) ===")
print(hyb_summary.to_string(index=False))

# 2. Compute exact crossover frame counts N* vs CPU (4T)
print("\n=== CROSSOVER BREAK-EVEN ANALYSIS (vs CPU 4T) ===")
crossover_rows = []
for m in ["SqueezeNet 1.1", "MobileNetV2"]:
    sub = summary[summary["model_name"] == m].set_index("backend")
    c_cold = sub.loc["CPU (4T)", "cold_total_ms"]
    c_step = sub.loc["CPU (4T)", "steady_med"]
    for gpu in ["RTX 3050 (DML-0)", "Intel UHD (DML-1)"]:
        g_cold = sub.loc[gpu, "cold_total_ms"]
        g_step = sub.loc[gpu, "steady_med"]
        if c_step > g_step:
            n_star = int(np.ceil((g_cold - c_cold) / (c_step - g_step)))
        else:
            n_star = -1  # Never crosses over
        print(f"{m} | {gpu}: Cold={g_cold:.1f}ms, Steady={g_step:.3f}ms vs CPU(4T) Cold={c_cold:.1f}ms, Steady={c_step:.3f}ms => N* = {n_star}")
        crossover_rows.append({"model": m, "gpu": gpu, "n_star": n_star})

        # Mann-Whitney U test on TTFI: Pure Cold GPU vs Hybrid Dispatcher
        raw_gpu = cold[(cold["model_name"] == m) & (cold["backend"] == gpu)]
        raw_gpu_ttfi = raw_gpu["session_create_ms"] + raw_gpu["first_inf_ms"]
        raw_hyb_ttfi = hyb[(hyb["model_name"] == m) & (hyb["gpu_name"] == gpu)]["ttfi_ms"]
        stat, pval = mannwhitneyu(raw_gpu_ttfi, raw_hyb_ttfi, alternative="greater")
        print(f"   Mann-Whitney U (Cold GPU TTFI > Hybrid TTFI): U={stat:.1f}, p={pval:.3e}")

# 3. Figure 1: Cold-Start Breakdown (Log-scale grouped bars)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
order = ["CPU (1T)", "CPU (4T)", "Intel UHD (DML-1)", "RTX 3050 (DML-0)"]
colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

for ax, m in zip(axes, ["SqueezeNet 1.1", "MobileNetV2"]):
    sub = summary[summary["model_name"] == m].set_index("backend").reindex(order)
    x = np.arange(len(order))
    b1 = ax.bar(x - 0.18, sub["cold_total_ms"], width=0.34, label="Cold Start (Init + Frame 1)", color="#2b5c8f")
    b2 = ax.bar(x + 0.18, sub["steady_med"], width=0.34, label="Steady-State Median", color="#f28e2b")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["CPU\n(1T)", "CPU\n(4T)", "Intel\niGPU", "RTX\n3050"])
    ax.set_title(m)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

axes[0].set_ylabel("Latency (ms, log scale)")
axes[1].legend(loc="upper left")
plt.tight_layout()
plt.savefig("paper/figures/fig1_coldstart_breakdown.pdf")
plt.savefig("paper/figures/fig1_coldstart_breakdown.png")
plt.close()

# 4. Figure 2: Cumulative Latency Crossover & Hybrid Trajectory
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
frames = np.arange(1, 1001)

for ax, m in zip(axes, ["SqueezeNet 1.1", "MobileNetV2"]):
    sub = summary[summary["model_name"] == m].set_index("backend")
    hsub = hyb_summary[hyb_summary["model_name"] == m].set_index("gpu_name")

    for backend, style, col in [
        ("CPU (4T)", "--", "#2ca02c"),
        ("Intel UHD (DML-1)", "-.", "#ff7f0e"),
        ("RTX 3050 (DML-0)", ":", "#d62728")
    ]:
        c0 = sub.loc[backend, "cold_total_ms"]
        dt = sub.loc[backend, "steady_med"]
        cum = c0 + (frames - 1) * dt
        ax.plot(frames, cum / 1000.0, linestyle=style, color=col, linewidth=1.8, label=backend)

    # Hybrid trajectory using Intel UHD (fast switch) or RTX 3050
    for gpu, col in [("Intel UHD (DML-1)", "#1f77b4"), ("RTX 3050 (DML-0)", "#9467bd")]:
        sw = int(hsub.loc[gpu, "switch_frame_med"])
        ttfi = hsub.loc[gpu, "ttfi_med"]
        c_dt = hsub.loc[gpu, "cpu_phase_med"]
        g_dt = hsub.loc[gpu, "gpu_phase_med"]
        cum_h = []
        t_acc = ttfi
        for f in frames:
            if f == 1:
                cum_h.append(t_acc)
            elif f <= sw:
                t_acc += c_dt
                cum_h.append(t_acc)
            else:
                t_acc += g_dt
                cum_h.append(t_acc)
        ax.plot(frames, np.array(cum_h) / 1000.0, color=col, linewidth=2.0, label=f"Hybrid ({gpu.split()[0]})")

    ax.set_title(m)
    ax.set_xlabel("Completed Inference Frames (N)")
    ax.grid(True, linestyle="--", alpha=0.4)

axes[0].set_ylabel("Cumulative Wall Time (seconds)")
axes[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig("paper/figures/fig2_cumulative_crossover.pdf")
plt.savefig("paper/figures/fig2_cumulative_crossover.png")
plt.close()

# 5. Figure 3: TTFI Reduction (Cold GPU vs Hybrid Dispatcher)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
categories = ["SqueezeNet\n(Intel iGPU)", "SqueezeNet\n(RTX 3050)", "MobileNetV2\n(Intel iGPU)", "MobileNetV2\n(RTX 3050)"]
cold_vals = []
hyb_vals = []
for m in ["SqueezeNet 1.1", "MobileNetV2"]:
    sub = summary[summary["model_name"] == m].set_index("backend")
    hsub = hyb_summary[hyb_summary["model_name"] == m].set_index("gpu_name")
    for gpu in ["Intel UHD (DML-1)", "RTX 3050 (DML-0)"]:
        cold_vals.append(sub.loc[gpu, "cold_total_ms"])
        hyb_vals.append(hsub.loc[gpu, "ttfi_med"])

x = np.arange(len(categories))
b1 = ax.bar(x - 0.18, cold_vals, width=0.34, label="Pure Cold DirectML (TTFI)", color="#d62728")
b2 = ax.bar(x + 0.18, hyb_vals, width=0.34, label="Proposed Hybrid Dispatcher (TTFI)", color="#1f77b4")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Time-to-First-Inference (ms, log scale)")
ax.set_title("Eliminating Cold-Start Stalls via Asynchronous CPU-to-GPU Warm-Up")
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h * 1.15, f"{h:.0f} ms", ha="center", va="bottom", fontsize=7.5)
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h * 1.15, f"{h:.1f} ms", ha="center", va="bottom", fontsize=7.5)
ax.legend()
plt.tight_layout()
plt.savefig("paper/figures/fig3_hybrid_ttfi_comparison.pdf")
plt.savefig("paper/figures/fig3_hybrid_ttfi_comparison.png")
plt.close()

print("\nFIGURES SAVED TO paper/figures/ successfully!")
