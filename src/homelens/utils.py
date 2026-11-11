"""Small helpers: results registry, plotting style, RMSE."""
import json

import matplotlib as mpl
import numpy as np

from .config import FIG_DIR, RESULTS_PATH

PALETTE = {"blue": "#1f4e79", "orange": "#d9822b", "green": "#2e7d5b", "red": "#b23a48",
           "grey": "#6b7280", "teal": "#1b8a9a", "purple": "#6a4c93"}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 170, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "font.family": "DejaVu Sans", "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE.values())),
    })


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"[figure saved] {path.name}")


def record(key, value):
    """Persist a number/string so the PDF builder can quote real results."""
    data = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    data[key] = value
    RESULTS_PATH.write_text(json.dumps(data, indent=1, default=float))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))
