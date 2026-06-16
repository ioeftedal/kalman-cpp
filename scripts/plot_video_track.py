#!/usr/bin/env python3
"""Plot video tracking results from the CSV log.

Usage:
    uv run python3 scripts/plot_video_track.py output/video_track_log.csv
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: str) -> dict[str, np.ndarray]:
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    data = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot video tracking results")
    parser.add_argument("csv", nargs="?", default="output/video_track_log.csv",
                        help="Path to CSV log")
    parser.add_argument("--output", default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    try:
        data = read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: {args.csv} not found. Run ./video_track first.")
        return 1

    time_s = data["time"]
    is_measured = (data["meas_u"] != 0) | (data["meas_v"] != 0)

    # Position error when measurements exist
    err_u = data["est_u"] - data["meas_u"]
    err_v = data["est_v"] - data["meas_v"]
    pos_err = np.sqrt(err_u ** 2 + err_v ** 2)
    pos_err[~is_measured] = np.nan

    # 3-sigma bounds
    sigma_u = 3.0 * np.sqrt(data["P_uu"])
    sigma_v = 3.0 * np.sqrt(data["P_vv"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Video Tracking — UKF Pixel CV", fontsize=14)

    # 1. Trajectory
    ax = axes[0, 0]
    meas = ax.scatter(data["meas_u"][is_measured],
                      data["meas_v"][is_measured],
                      c=time_s[is_measured], cmap="Blues", s=8,
                      label="Detection", alpha=0.7)
    ax.plot(data["est_u"], data["est_v"], "r-", lw=1.5, label="UKF estimate")
    ax.scatter(data["est_u"][0], data["est_v"][0], c="green", s=80,
               marker="o", label="Start", zorder=5)
    ax.scatter(data["est_u"][-1], data["est_v"][-1], c="red", s=80,
               marker="x", label="End", zorder=5)
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.set_title("Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 2. Position error
    ax = axes[0, 1]
    ax.plot(time_s[is_measured], pos_err[is_measured], "b.", ms=2, alpha=0.5)
    valid = ~np.isnan(pos_err)
    if np.any(valid):
        mean_err = np.nanmean(pos_err)
        ax.axhline(mean_err, color="r", ls="--", lw=1,
                   label=f"Mean: {mean_err:.1f} px")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (px)")
    ax.set_title("Position Error vs Detection")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Velocity estimates
    ax = axes[1, 0]
    ax.plot(time_s, data["est_vu"], "b-", lw=1, label="vu (u-velocity)")
    ax.plot(time_s, data["est_vv"], "r-", lw=1, label="vv (v-velocity)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (px/s)")
    ax.set_title("Velocity Estimates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. NEES (filter consistency)
    ax = axes[1, 1]
    ax.plot(time_s, data["nees"], "k.", ms=2, alpha=0.4)
    ax.axhline(4.0, color="r", ls="--", lw=1,
               label="Expected (4 DOF)")
    ax.axhline(13.277, color="orange", ls=":", lw=1,
               label="χ² 99% (13.28)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_title("Normalised Estimation Error Squared")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = args.output or "output/video_track_plot.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
