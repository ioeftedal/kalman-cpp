import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import read_csv


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/sim_output.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Run the simulator first.")
        sys.exit(1)

    d = read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: 2D trajectory ---
    ax1.plot(d["true_x"], d["true_y"], "k-", linewidth=1.5, label="True", alpha=0.8)
    ax1.scatter(d["meas_x"], d["meas_y"], s=3, c="tab:blue", alpha=0.4, label="Noisy meas.")
    ax1.plot(d["est_x"], d["est_y"], "r-", linewidth=1.2, label="UKF estimate", alpha=0.9)
    # Mark start/end
    ax1.scatter(d["true_x"][0], d["true_y"][0], marker="s", c="green", s=60, zorder=5, label="Start")
    ax1.scatter(d["true_x"][-1], d["true_y"][-1], marker="X", c="red", s=60, zorder=5, label="End")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("2D Trajectory — True vs Estimated")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # --- Right: position error vs time ---
    err = np.sqrt((d["true_x"] - d["est_x"]) ** 2 + (d["true_y"] - d["est_y"]) ** 2)
    meas_err = np.sqrt((d["true_x"] - d["meas_x"]) ** 2 + (d["true_y"] - d["meas_y"]) ** 2)
    ax2.plot(d["time"], err, "r-", linewidth=1, label=r"$||\hat{x} - x_{\mathrm{true}}||$")
    ax2.plot(d["time"], meas_err, "b-", linewidth=0.5, alpha=0.5, label=r"$||z - x_{\mathrm{true}}||$ (meas.)")
    ax2.axhline(y=np.mean(err), color="r", ls="--", linewidth=0.7, alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Position Error Over Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(csv_path).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
