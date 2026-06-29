#!/usr/bin/env python3
"""Extract lateral control data from rlogs to CSV/Parquet.

Replaces the complex 3-file pipeline (lat.py -> lat_to_csv.py / lat_to_csv_torquennd.py)
with a single script that reads rlogs and emits one row per controlsState message.

Usage:
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.parquet --format parquet
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv --temporal
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Temporal offsets matching nnlc.py's past_times and future_times
PAST_TIMES = [-0.3, -0.2, -0.1]
FUTURE_TIMES = [0.3, 0.6, 1.0, 1.5]

COLUMNS = [
    "timestamp",
    "v_ego",
    "a_ego",
    "steering_angle_deg",
    "steering_rate_deg",
    "steering_torque",
    "steering_pressed",
    "standstill",
    "desired_curvature",
    "curvature",
    "active",
    "lateral_control_type",
    "actual_lateral_accel",
    "desired_lateral_accel",
    "torque_output",
    "saturated",
    "roll",
    "lane_change_state",
]


def find_rlogs(input_dir):
    """Find all rlog files in the input directory."""
    patterns = ["**/rlog.zst", "**/rlog.bz2", "**/rlog"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=True))
    return sorted(set(files))


def extract_segment(rlog_path):
    """Extract lateral data rows from a single rlog file.

    Follows the message iteration pattern from openpilot's
    measure_steering_accuracy.py — accumulate messages by type in a state dict,
    then emit a row when controlsState arrives.
    """
    from nnlc_tools.logreader import LogReader

    rows = []
    sm = {}

    try:
        lr = LogReader(rlog_path, sort_by_time=True)
    except Exception as e:
        print(f"  WARNING: Could not open {rlog_path}: {e}")
        return rows

    unknown_events = 0
    try:
        for msg in lr:
            try:
                msg_type = msg.which()
            except Exception:
                # Logs can contain newer Event union variants that are absent
                # from the bundled cereal schema. They are unrelated to the
                # signals extracted here, so skip them instead of discarding
                # the remainder of the segment.
                unknown_events += 1
                continue

            if msg_type == "carState":
                sm["carState"] = msg.carState
            elif msg_type == "controlsState":
                sm["controlsState"] = msg.controlsState
            elif msg_type == "selfdriveState":
                sm["selfdriveState"] = msg.selfdriveState
            elif msg_type == "liveParameters":
                sm["liveParameters"] = msg.liveParameters
            elif msg_type == "modelV2":
                sm["modelV2"] = msg.modelV2

            # Emit a row on each controlsState when we have carState too
            if msg_type == "controlsState" and "carState" in sm:
                cs = sm["carState"]
                ctrl = sm["controlsState"]

                timestamp = msg.logMonoTime / 1e9

                # Determine lateral control type and extract type-specific fields
                lat_state = ctrl.lateralControlState
                lat_type = lat_state.which()

                actual_lat_accel = float("nan")
                desired_lat_accel = float("nan")
                torque_output = float("nan")
                saturated = False

                if lat_type == "torqueState":
                    ts = lat_state.torqueState
                    actual_lat_accel = ts.actualLateralAccel
                    desired_lat_accel = ts.desiredLateralAccel
                    torque_output = ts.output
                    saturated = ts.saturated
                elif lat_type == "pidState":
                    ps = lat_state.pidState
                    torque_output = ps.output
                    saturated = ps.saturated

                # active moved from controlsState to selfdriveState in newer openpilot
                if "selfdriveState" in sm:
                    active = sm["selfdriveState"].active
                else:
                    active = getattr(ctrl, "active", getattr(ctrl, "activeDEPRECATED", False))

                roll = sm["liveParameters"].roll if "liveParameters" in sm else float("nan")

                lane_change_state = 0
                if "modelV2" in sm:
                    try:
                        lane_change_state = int(sm["modelV2"].meta.laneChangeState)
                    except (AttributeError, ValueError, TypeError):
                        pass

                row = [
                    timestamp,
                    cs.vEgo,
                    cs.aEgo,
                    cs.steeringAngleDeg,
                    cs.steeringRateDeg,
                    cs.steeringTorque,
                    cs.steeringPressed,
                    cs.standstill,
                    ctrl.desiredCurvature,
                    ctrl.curvature,
                    active,
                    lat_type,
                    actual_lat_accel,
                    desired_lat_accel,
                    torque_output,
                    saturated,
                    roll,
                    lane_change_state,
                ]
                rows.append(row)
        if unknown_events:
            print(f"  WARNING: Skipped {unknown_events} unknown event(s) in {rlog_path}")
    except Exception as e:
        print(f"  WARNING: Error processing {rlog_path}: {e}")

    return rows


def add_temporal_columns(df):
    """Add lagged/lead columns for temporal model training.

    Adds columns at offsets matching nnlc.py's past_times [-0.3, -0.2, -0.1]
    and future_times [0.3, 0.6, 1.0, 1.5].
    """
    dt = 0.01  # controlsState runs at 100Hz
    temporal_cols = ["actual_lateral_accel", "desired_lateral_accel", "roll"]

    for offset in PAST_TIMES + FUTURE_TIMES:
        shift_frames = int(round(offset / dt))
        suffix = f"_t{offset:+.1f}".replace(".", "").replace("+", "p").replace("-", "m")

        for col in temporal_cols:
            if col in df.columns:
                df[f"{col}{suffix}"] = df[col].shift(-shift_frames)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract lateral control data from rlogs to CSV/Parquet.",
    )
    parser.add_argument("input", help="Directory containing rlog files")
    parser.add_argument("-o", "--output", default="lateral_data.csv",
                        help="Output file path (default: lateral_data.csv)")
    parser.add_argument("--format", choices=["csv", "parquet"], default=None,
                        help="Output format (default: inferred from extension)")
    parser.add_argument("--temporal", action="store_true",
                        help="Add temporal lag/lead columns for NNLC training")
    parser.add_argument("--filter-overrides", action="store_true",
                        help="Drop rows where driver overrides (steering_pressed=True)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    rlog_files = find_rlogs(args.input)
    if not rlog_files:
        print(f"ERROR: No rlog files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(rlog_files)} rlog files")

    all_rows = []
    for rlog_path in tqdm(rlog_files, desc="Processing rlogs"):
        rows = extract_segment(rlog_path)
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: No data extracted from any rlog files")
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    print(f"Extracted {len(df)} rows")

    if args.filter_overrides and "steering_pressed" in df.columns:
        before = len(df)
        df = df[~df["steering_pressed"].astype(bool)]
        dropped = before - len(df)
        print(f"Filtered {dropped} override rows ({dropped / before:.1%} of data)")

    if args.temporal:
        print("Adding temporal columns...")
        df = add_temporal_columns(df)

    # Determine output format
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".parquet"):
            fmt = "parquet"
        else:
            fmt = "csv"

    if fmt == "parquet":
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"Saved to {args.output} ({fmt})")


if __name__ == "__main__":
    main()
