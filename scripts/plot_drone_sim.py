#!/usr/bin/env python3
"""Plot dual CV/CA drone simulation output — 2D pixel-space comparison."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import read_csv


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "output/drone_sim_output"
    qc_cv = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
    qc_ca = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
    cv_path = f"{prefix}_cv.csv"
    ca_path = f"{prefix}_ca.csv"

    if not Path(cv_path).exists() or not Path(ca_path).exists():
        print(f"Error: {cv_path} or {ca_path} not found. Run ./drone_sim first.")
        sys.exit(1)

    cv = read_csv(cv_path)
    ca = read_csv(ca_path)

    img_w, img_h = 3840, 2160

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # --- 1. Image-plane trajectory ---
    ax1 = fig.add_subplot(gs[0, 0])
    true_all = cv["true_u"]  # same for both
    ax1.plot(cv["true_u"], cv["true_v"], "k-", lw=1.5, alpha=0.7, label="True")

    cv_f = cv["type"] == 0.0
    ca_f = ca["type"] == 0.0
    ax1.plot(cv["est_u"][cv_f], cv["est_v"][cv_f], "r-", lw=1.0, alpha=0.6, label="CV estimate")
    ax1.plot(ca["est_u"][ca_f], ca["est_v"][ca_f], "b-", lw=1.0, alpha=0.6, label="CA estimate")

    cv_p = cv["type"] == 1.0
    ca_p = ca["type"] == 1.0
    if cv_p.any():
        ax1.plot(cv["est_u"][cv_p], cv["est_v"][cv_p], "r--", lw=2.0, alpha=0.7, label="CV prediction")
    if ca_p.any():
        ax1.plot(ca["est_u"][ca_p], ca["est_v"][ca_p], "b--", lw=2.0, alpha=0.7, label="CA prediction")
    if cv_p.any():
        ax1.plot(cv["true_u"][cv_p], cv["true_v"][cv_p], "k:", lw=1.0, alpha=0.3, label="Future truth")

    ax1.scatter(cv["true_u"][0], cv["true_v"][0], marker="s", c="green", s=60, zorder=5)
    ax1.add_patch(plt.Rectangle((0, 0), img_w, img_h, fill=False,
                                ec="gray", ls="--", lw=0.5))
    ax1.set_xlabel("u (pixels)")
    ax1.set_ylabel("v (pixels)")
    ax1.set_title("Image-plane Trajectory")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_aspect("equal")
    ax1.set_xlim(-img_w * 0.05, img_w * 1.05)
    ax1.set_ylim(img_h * 1.05, -img_h * 0.05)
    ax1.grid(True, alpha=0.3)

    # --- 2. Position error vs time ---
    ax2 = fig.add_subplot(gs[0, 1])
    for d, name, color in [(cv, "CV", "r"), (ca, "CA", "b")]:
        mask = d["type"] == 0.0
        err = np.sqrt((d["true_u"][mask] - d["est_u"][mask])**2 +
                      (d["true_v"][mask] - d["est_v"][mask])**2)
        ax2.plot(d["time"][mask], err, color=color, lw=0.8, alpha=0.7, label=f"{name}")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position error (px)")
    ax2.set_title("Filter Position Error")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 3. Cumulative error CDF ---
    ax3 = fig.add_subplot(gs[0, 2])
    for d, name, color in [(cv, "CV", "r"), (ca, "CA", "b")]:
        mask = d["type"] == 0.0
        err = np.sqrt((d["true_u"][mask] - d["est_u"][mask])**2 +
                      (d["true_v"][mask] - d["est_v"][mask])**2)
        err_s = np.sort(err)
        cdf = np.arange(1, len(err_s)+1) / len(err_s)
        ax3.plot(err_s, cdf, color=color, lw=1.5, label=f"{name}")
        ax3.axvline(x=np.mean(err), color=color, ls="--", lw=0.7, alpha=0.5)

    ax3.set_xlabel("Position error (px)")
    ax3.set_ylabel("CDF")
    ax3.set_title("Error Distribution")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- 4. 3-sigma normalised error ---
    ax4 = fig.add_subplot(gs[1, 0])
    for d, name, color in [(cv, "CV", "r"), (ca, "CA", "b")]:
        mask = d["type"] == 0.0
        err = np.sqrt((d["true_u"][mask] - d["est_u"][mask])**2 +
                      (d["true_v"][mask] - d["est_v"][mask])**2)
        three_sigma = 3.0 * np.sqrt(d["P_uu"][mask] + d["P_vv"][mask])
        ax4.plot(d["time"][mask], err / three_sigma, color=color, lw=0.8, alpha=0.7,
                 label=f"{name}")
    ax4.axhline(y=1.0, color="gray", ls="--", lw=0.5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Error / 3σ")
    ax4.set_title("Normalised Error")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- 5. Prediction error ---
    ax5 = fig.add_subplot(gs[1, 1])
    for d, name, color in [(cv, "CV", "r"), (ca, "CA", "b")]:
        mask = d["type"] == 1.0
        if not mask.any():
            continue
        pred_time = d["time"][mask] - d["time"][mask][0]
        err = np.sqrt((d["true_u"][mask] - d["est_u"][mask])**2 +
                      (d["true_v"][mask] - d["est_v"][mask])**2)
        ax5.plot(pred_time, err, color=color, lw=1.5, label=f"{name}")
        three_sigma = 3.0 * np.sqrt(d["P_uu"][mask] + d["P_vv"][mask])
        ax5.plot(pred_time, three_sigma, color=color, ls="--", lw=0.7, alpha=0.4)

    ax5.set_xlabel("Look-ahead time (s)")
    ax5.set_ylabel("Position error (px)")
    ax5.set_title("Async Prediction Error")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- 6. Summary table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    def rmse(d):
        m = d["type"] == 0.0
        return math.sqrt(np.mean((d["true_u"][m] - d["est_u"][m])**2 +
                                 (d["true_v"][m] - d["est_v"][m])**2))

    def pred_auc(d):
        m = d["type"] == 1.0
        if not m.any():
            return 0.0
        err = np.sqrt((d["true_u"][m] - d["est_u"][m])**2 +
                      (d["true_v"][m] - d["est_v"][m])**2)
        dt_pred = d["time"][m][1] - d["time"][m][0] if sum(m) > 1 else 0
        return float(np.sum(err) * dt_pred)

    cv_rmse = rmse(cv)
    ca_rmse = rmse(ca)
    cv_auc = pred_auc(cv)
    ca_auc = pred_auc(ca)

    rows = [
        ["Metric", "CV", "CA", "Better"],
        ["Filter RMSE (px)", f"{cv_rmse:.2f}", f"{ca_rmse:.2f}",
         "CV" if cv_rmse < ca_rmse else "CA"],
        ["Pred AUC (px·s)", f"{cv_auc:.0f}", f"{ca_auc:.0f}",
         "CV" if cv_auc < ca_auc else "CA"],
        ["qc", f"{qc_cv:.0f}", f"{qc_ca:.0f}", ""],
        ["State dim", "4", "6", ""],
        ["Model", "CV", "CA", ""],
    ]
    table = ax6.table(cellText=rows, loc="center", cellLoc="center",
                      colWidths=[0.18, 0.12, 0.12, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", weight="bold")
        elif c == 3:
            cell.set_facecolor("#e8f5e9" if "CA" in cell.get_text().get_text() else "#fff3e0")
    ax6.set_title("Summary", fontsize=11, pad=20)

    out = Path(f"{prefix}_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
