import argparse
import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

# Allow running from repo root without installation
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

sys.path.append('/Users/lorenzo/Minerva/fork_reweighting')

from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.analysis import transform_momentum_to_reaction_frame



DERIVED_VARIABLES = {
    "muon_pt": "sqrt(px^2+py^2)",
    "muon_p": "sqrt(px^2+py^2+pz^2)",
    "muon_cos_theta": "pz/sqrt(px^2+py^2+pz^2)",
    "muon_py_reaction": "py in reaction frame (px=0)",
}

BASIC_VARIABLE_MAP = {
    "leading_muon_px": "leading_muon_px",
    "leading_muon_py": "leading_muon_py",
    "leading_muon_pz": "leading_muon_pz",
    "leading_muon_E": "leading_muon_E",
    "leading_muon_KE": "leading_muon_KE",
}

# Map our friendly names to NuisanceFlatTree expressions
NUISANCE_EXPR = {
    "leading_muon_px": "leading_muon_px",
    "leading_muon_py": "leading_muon_py",
    "leading_muon_pz": "leading_muon_pz",
    "leading_muon_E": "leading_muon_E",
    "leading_muon_KE": "leading_muon_KE",
}


def parse_bins(bins_arg: str):
    if bins_arg is None:
        return 60
    try:
        return int(bins_arg)
    except ValueError:
        edges = [float(x) for x in bins_arg.split(',') if x.strip() != ""]
        if len(edges) < 2:
            raise ValueError("Bin edges string must have at least two comma-separated numbers")
        return np.array(edges, dtype=float)


def load_muon_var(tree: NuisanceFlatTree, var_name: str, mask: Optional[np.ndarray]):
    var_name = var_name.strip()
    if var_name in NUISANCE_EXPR:
        arr = tree.get_event_variable(NUISANCE_EXPR[var_name], mask=mask)
        return ak.to_numpy(ak.fill_none(arr, np.nan))

    # Derived kinematics
    px = tree.get_event_variable("leading_muon_px", mask=mask)
    py = tree.get_event_variable("leading_muon_py", mask=mask)
    pz = tree.get_event_variable("leading_muon_pz", mask=mask)
    px = ak.to_numpy(ak.fill_none(px, np.nan))
    py = ak.to_numpy(ak.fill_none(py, np.nan))
    pz = ak.to_numpy(ak.fill_none(pz, np.nan))

    if var_name == "muon_pt":
        return np.sqrt(px * px + py * py)
    if var_name == "muon_p":
        return np.sqrt(px * px + py * py + pz * pz)
    if var_name == "muon_cos_theta":
        p = np.sqrt(px * px + py * py + pz * pz)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, pz / p, np.nan)
    if var_name == "muon_py_reaction":
        df = pd.DataFrame({
            "leading_muon_px": px,
            "leading_muon_py": py,
            "leading_muon_pz": pz,
        })
        rotated = transform_momentum_to_reaction_frame(df, selector_lepton="leading_muon", particle_names=[])
        return rotated["leading_muon_py"].to_numpy()

    raise ValueError(f"Unsupported variable '{var_name}'.")


def build_mask(tree: NuisanceFlatTree, mode_filter: Optional[List[int]], max_events: Optional[int]):
    base_mask = np.full(len(tree._flattree_vars), True)
    if mode_filter:
        modes = np.array(tree.get_event_variable("Mode"))
        mode_mask = np.isin(modes, mode_filter)
        base_mask = base_mask & mode_mask
    if max_events is not None:
        base_mask_indices = np.nonzero(base_mask)[0]
        keep = base_mask_indices[:max_events]
        mask = np.full(len(tree._flattree_vars), False)
        mask[keep] = True
        base_mask = mask
    return base_mask


def summarize_events(tree: NuisanceFlatTree, var: np.ndarray, event_indices: np.ndarray, top_n: int):
    if top_n <= 0:
        return []
    if len(var) == 0 or len(event_indices) == 0:
        return []

    weights_all = tree.get_weight()
    enu_all = ak.to_numpy(ak.fill_none(tree.get_event_variable("Enu_true"), np.nan))
    mode_all = tree.get_mode()
    px_all = ak.to_numpy(ak.fill_none(tree.get_event_variable("leading_muon_px"), np.nan))
    py_all = ak.to_numpy(ak.fill_none(tree.get_event_variable("leading_muon_py"), np.nan))
    pz_all = ak.to_numpy(ak.fill_none(tree.get_event_variable("leading_muon_pz"), np.nan))

    finite_mask = np.isfinite(var)
    if not np.any(finite_mask):
        return []
    local_indices = np.nonzero(finite_mask)[0]
    top_local = local_indices[np.argsort(var[finite_mask])[-top_n:]][::-1]

    rows = []
    for local_idx in top_local:
        global_idx = int(event_indices[local_idx])
        rows.append({
            "idx": global_idx,
            "var": var[local_idx],
            "weight": weights_all[global_idx] if global_idx < len(weights_all) else np.nan,
            "mode": mode_all[global_idx] if global_idx < len(mode_all) else np.nan,
            "Enu_true": enu_all[global_idx] if global_idx < len(enu_all) else np.nan,
            "px": px_all[global_idx] if global_idx < len(px_all) else np.nan,
            "py": py_all[global_idx] if global_idx < len(py_all) else np.nan,
            "pz": pz_all[global_idx] if global_idx < len(pz_all) else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Quick muon-kinematics plots from a NUISANCE flat tree.")
    parser.add_argument("--input", "-i", required=True, help="Path to NUISANCE flat tree ROOT file.")
    parser.add_argument("--variable", "-v", default="muon_pt", help="Variable to plot: leading_muon_px, leading_muon_py, leading_muon_pz, leading_muon_E, leading_muon_KE, muon_pt, muon_p, muon_cos_theta, muon_py_reaction.")
    parser.add_argument("--bins", help="Either an integer (number of bins) or a comma-separated list of bin edges.")
    parser.add_argument("--range", dest="range_str", help="If --bins is an int, optionally specify min,max range (e.g. '0,3').")
    parser.add_argument("--mode", nargs="*", type=int, help="Filter to specific interaction Mode codes (space separated).")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on number of events after filtering.")
    parser.add_argument("--output", "-o", help="Output image file. Defaults to <input>_<variable>.png in the same folder.")
    parser.add_argument("--no-weights", action="store_true", help="Ignore event weights and plot unweighted counts.")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis.")
    parser.add_argument("--top", type=int, default=0, help="Print the top-N events by the plotted variable.")

    args = parser.parse_args()

    bins = parse_bins(args.bins)
    range_tuple = None
    if args.range_str is not None:
        parts = [float(x) for x in args.range_str.split(',') if x.strip() != ""]
        if len(parts) == 2:
            range_tuple = (parts[0], parts[1])

    tree = NuisanceFlatTree(args.input)
    mask = build_mask(tree, args.mode, args.max_events)
    selected_indices = np.nonzero(mask)[0]
    var = load_muon_var(tree, args.variable, mask)

    weights = None if args.no_weights else tree.get_weight()[mask]

    finite = np.isfinite(var)
    var = var[finite]
    selected_indices = selected_indices[finite]
    if weights is not None:
        weights = weights[finite]
    if var.size == 0:
        print("No finite entries to plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(var, bins=bins, range=range_tuple, weights=weights, histtype="stepfilled", alpha=0.65, color="tab:blue")
    plt.xlabel(args.variable)
    plt.ylabel("Events (weighted)" if weights is not None else "Events")
    plt.grid(True, alpha=0.3)
    if args.logy:
        plt.yscale("log")

    out_path = args.output
    if out_path is None:
        out_path = pathlib.Path(args.input).with_suffix("")
        out_path = str(out_path) + f"_{args.variable}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

    if args.top > 0:
        summaries = summarize_events(tree, var, selected_indices, args.top)
        if summaries:
            print(f"Top {len(summaries)} events by {args.variable}:")
            for row in summaries:
                print(
                    f"idx={row['idx']:6d} var={row['var']:8.4g} w={row['weight']:8.4g} mode={row['mode']:<4d} "
                    f"Enu={row['Enu_true']:6.3f} px={row['px']:7.3f} py={row['py']:7.3f} pz={row['pz']:7.3f}"
                )
        else:
            print("No events to summarize.")


if __name__ == "__main__":
    main()
