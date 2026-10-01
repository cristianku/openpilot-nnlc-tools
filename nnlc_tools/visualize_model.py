#!/usr/bin/env python3
"""Visualize trained NNLC model predictions against actual data.

Generates two plot types matching sunnypilot PR validation style:
- lat_accel_vs_torque: one subplot per speed bin, data scatter + model curve
- torque_vs_speed: one subplot per lat_accel bin, data scatter + model curve

Usage:
  python -m nnlc_tools.visualize_model model.json data.csv -o output/
"""

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# [nnlc contract] - START
# NNTorqueModel consumes inputs by position and does not read input_vars.
NNLC_INPUT_COLUMNS = [
    "v_ego", "actual_lateral_accel", "lateral_jerk", "roll",
    "actual_lateral_accel_tm03", "actual_lateral_accel_tm02", "actual_lateral_accel_tm01",
    "actual_lateral_accel_tp03", "actual_lateral_accel_tp06", "actual_lateral_accel_tp10",
    "actual_lateral_accel_tp15", "roll_tm03", "roll_tm02", "roll_tm01", "roll_tp03",
    "roll_tp06", "roll_tp10", "roll_tp15",
]
# [nnlc contract] - END


# Activation functions
def sigmoid(x):
    # Match Sunnypilot's safe_exp, including the clamped negative tail.
    return 1.0 / (1.0 + np.exp(np.minimum(-x, 11.0)))


def identity(x):
    return x


def tanh(x):
    return np.tanh(x)


def leakyrelu(x):
    return np.where(x > 0, x, 0.01 * x)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "σ": sigmoid,
    "identity": identity,
    "tanh": tanh,
    "leakyrelu": leakyrelu,
}


class NNModel:
    """Lightweight NN model loaded from JSON for inference."""

    def __init__(self, params_file):
        with open(params_file) as f:
            params = json.load(f)

        self.input_size = params["input_size"]
        self.output_size = params["output_size"]
        self.input_mean = np.array(params["input_mean"], dtype=np.float32).T.flatten()
        self.input_std = np.array(params["input_std"], dtype=np.float32).T.flatten()
        self.input_vars = params.get("input_vars", [])

        self.layers = []
        for layer_params in params["layers"]:
            W_key = next(k for k in layer_params if k.endswith("_W"))
            b_key = next(k for k in layer_params if k.endswith("_b"))
            W = np.array(layer_params[W_key], dtype=np.float32).T
            b = np.array(layer_params[b_key], dtype=np.float32).T.flatten()
            act_name = layer_params["activation"]
            act_fn = ACTIVATIONS.get(act_name)
            if act_fn is None:
                raise ValueError(f"Unknown activation: {act_name}")
            self.layers.append((W, b, act_fn))

    def forward(self, x):
        """Forward pass. x shape: (batch, input_size)."""
        for W, b, act in self.layers:
            x = act(x @ W + b)
        return x

    def predict(self, x):
        """Normalize inputs and run forward pass."""
        x_norm = (x - self.input_mean) / self.input_std
        return self.forward(x_norm)

    def var_index(self, name):
        """Get index of an input variable by name."""
        return self.input_vars.index(name)

    def make_input_at_means(self, n=1):
        """Create input array filled with mean values (un-normalized space)."""
        return np.tile(self.input_mean, (n, 1))

    # [nnlc contract] - START
    def validate_nnlc(self):
        """Check positional compatibility and internal torque direction."""
        if self.input_size != 18 or self.output_size != 1 or self.input_vars != NNLC_INPUT_COLUMNS:
            raise ValueError(
                "Model inputs do not match current Sunnypilot NNLC: expected 18 inputs "
                "with lateral_jerk third, roll fourth, and one torque output. "
                "Re-extract and retrain; changing input_vars alone does not fix the weights."
            )
        if (self.input_mean.shape != (18,) or self.input_std.shape != (18,)
                or not np.all(np.isfinite(self.input_mean))
                or not np.all(np.isfinite(self.input_std)) or np.any(self.input_std <= 0)):
            raise ValueError("NNLC normalization must contain 18 finite means and positive standard deviations")
        if not self.layers or any(act not in (sigmoid, identity) for _, _, act in self.layers):
            raise ValueError("Current Sunnypilot NNLC supports only sigmoid and identity activations")

        accelerations = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        for speed in (10.0, 20.0, 30.0):
            inputs = np.zeros((len(accelerations), 18), dtype=np.float32)
            inputs[:, 0] = speed
            inputs[:, 1] = accelerations
            inputs[:, 4:11] = accelerations[:, None]
            outputs = self.predict(inputs)
            if (outputs.shape != (len(accelerations), 1)
                    or not np.all(np.isfinite(outputs)) or not np.all(np.diff(outputs[:, 0]) > 0)):
                raise ValueError(
                    f"Wrong NNLC internal torque direction at {speed:g} m/s: "
                    "torque must increase with lateral acceleration. Re-extract and retrain."
                )
    # [nnlc contract] - END


def load_data(csv_path):
    """Load training data CSV."""
    if csv_path.endswith(".parquet"):
        return pd.read_parquet(csv_path)
    return pd.read_csv(csv_path)


def filter_active(df):
    """Filter to active, non-standstill rows."""
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    return df[mask]


MS_TO_MPH = 2.23694


def plot_lat_accel_vs_torque(model, df, output_path):
    """Generate per-speed-bin plots: lat_accel (x) vs torque (y)."""
    df = filter_active(df)

    lat_col = "actual_lateral_accel" if "actual_lateral_accel" in df.columns else "desired_lateral_accel"
    if lat_col not in df.columns or "torque_output" not in df.columns:
        print("WARNING: Missing required columns for lat_accel_vs_torque plot.")
        return

    valid = df[["v_ego", lat_col, "torque_output"]].dropna()
    valid = valid.copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH

    speed_bins = list(range(0, 90, 10))
    n_bins = len(speed_bins)
    ncols = 3
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Lateral Accel vs Torque by Speed Bin", fontsize=14, fontweight="bold")

    lat_sweep = np.linspace(-3.5, 3.5, 200)
    lat_idx = model.var_index(lat_col) if lat_col in model.input_vars else model.var_index("actual_lateral_accel")

    for i, speed_lo in enumerate(speed_bins):
        speed_hi = speed_lo + 10
        ax = axes[i]

        bin_data = valid[(valid["speed_mph"] >= speed_lo) & (valid["speed_mph"] < speed_hi)]
        sc = ax.scatter(bin_data[lat_col], bin_data["torque_output"],
                        c=bin_data["speed_mph"], cmap="viridis",
                        vmin=speed_lo, vmax=speed_hi,
                        s=0.5, alpha=0.3, rasterized=True)
        fig.colorbar(sc, ax=ax, label="Speed (mph)", pad=0.02)

        # Model prediction curve at bin midpoint speed
        mid_speed_ms = (speed_lo + speed_hi) / 2.0 / MS_TO_MPH
        x_input = model.make_input_at_means(len(lat_sweep))
        x_input[:, model.var_index("v_ego")] = mid_speed_ms
        x_input[:, lat_idx] = lat_sweep
        # Set temporal lat_accel vars to the sweep value too
        for var in model.input_vars:
            if var.startswith("actual_lateral_accel_t") or var.startswith("desired_lateral_accel_t"):
                x_input[:, model.var_index(var)] = lat_sweep
        pred = model.predict(x_input).flatten()
        ax.plot(lat_sweep, pred, color="blue", linewidth=1.5)

        ax.set_title(f"{speed_lo}-{speed_hi} mph (n={len(bin_data)})", fontsize=10)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Lat Accel (m/s²)")
        ax.set_ylabel("Torque")

    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved lat_accel_vs_torque plot to {output_path}")
    plt.close()


def plot_torque_vs_speed(model, df, output_path):
    """Generate per-lat_accel-bin plots: speed (x) vs torque (y)."""
    df = filter_active(df)

    lat_col = "actual_lateral_accel" if "actual_lateral_accel" in df.columns else "desired_lateral_accel"
    if lat_col not in df.columns or "torque_output" not in df.columns:
        print("WARNING: Missing required columns for torque_vs_speed plot.")
        return

    valid = df[["v_ego", lat_col, "torque_output"]].dropna()
    valid = valid.copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH
    valid["abs_lat_accel"] = valid[lat_col].abs()
    valid["abs_torque"] = valid["torque_output"].abs()

    lat_bin_edges = np.arange(0.0, 3.2, 0.2)
    n_bins = len(lat_bin_edges) - 1
    ncols = 4
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Torque vs Speed by Lateral Accel Bin", fontsize=14, fontweight="bold")

    speed_sweep_mph = np.linspace(0, 90, 200)
    speed_sweep_ms = speed_sweep_mph / MS_TO_MPH
    lat_idx = model.var_index(lat_col) if lat_col in model.input_vars else model.var_index("actual_lateral_accel")

    for i in range(n_bins):
        lat_lo = lat_bin_edges[i]
        lat_hi = lat_bin_edges[i + 1]
        ax = axes[i]

        bin_data = valid[(valid["abs_lat_accel"] >= lat_lo) & (valid["abs_lat_accel"] < lat_hi)]
        sc = ax.scatter(bin_data["speed_mph"], bin_data["abs_torque"],
                        c=bin_data["abs_lat_accel"], cmap="viridis",
                        vmin=lat_lo, vmax=lat_hi,
                        s=0.5, alpha=0.3, rasterized=True)
        fig.colorbar(sc, ax=ax, label="Lat Accel (m/s²)", pad=0.02)

        # Model prediction curve at bin midpoint lat_accel
        mid_lat = (lat_lo + lat_hi) / 2.0
        x_input = model.make_input_at_means(len(speed_sweep_ms))
        x_input[:, model.var_index("v_ego")] = speed_sweep_ms
        x_input[:, lat_idx] = mid_lat
        for var in model.input_vars:
            if var.startswith("actual_lateral_accel_t") or var.startswith("desired_lateral_accel_t"):
                x_input[:, model.var_index(var)] = mid_lat
        pred = np.abs(model.predict(x_input).flatten())
        ax.plot(speed_sweep_mph, pred, color="blue", linewidth=1.5)

        ax.set_title(f"{lat_lo:.1f}-{lat_hi:.1f} m/s² (n={len(bin_data)})", fontsize=10)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("Speed (mph)")
        ax.set_ylabel("Torque")

    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved torque_vs_speed plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained NNLC model predictions against actual data.",
    )
    parser.add_argument("model", help="Trained model JSON file")
    parser.add_argument("data", help="Training data CSV/Parquet file")
    parser.add_argument("-o", "--output-dir", default="./output/",
                        help="Output directory for plots (default: ./output/)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    # [nnlc contract] - START
    try:
        model = NNModel(args.model)
        model.validate_nnlc()
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print("NNLC compatibility validation passed (offline; no vehicle validation)")
    # [nnlc contract] - END
    print(f"Loaded model: {model.input_size} inputs, {len(model.layers)} layers")
    print(f"  Input vars: {model.input_vars}")

    df = load_data(args.data)
    print(f"Loaded {len(df)} data rows")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_lat_accel_vs_torque(model, df, os.path.join(args.output_dir, "lat_accel_vs_torque.png"))
    plot_torque_vs_speed(model, df, os.path.join(args.output_dir, "torque_vs_speed.png"))


if __name__ == "__main__":
    main()
