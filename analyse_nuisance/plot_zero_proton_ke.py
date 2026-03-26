import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


import sys
sys.path.append('/Users/lorenzo/Minerva/fork_reweighting')

from BDTReweight.nuisance_flat_tree import NuisanceFlatTree

# Small helper to convert awkward arrays to numpy 1D arrays
# (keeps shape, drops jagged structure that should not be present here).
def _to_numpy(arr):
    return np.asarray(ak.to_numpy(arr))


def main():
    parser = argparse.ArgumentParser(description="Plot kinematics split by total_proton_KE == 0 vs > 0")
    parser.add_argument("-f", help="Path to NUISANCE FlatTree_VARS ROOT file", default = "/Users/lorenzo/cernbox/MINERVA_MC/target/flat_GENIE_G18_10b_02_11a_50M_CCQELike.root")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on number of events to read")
    parser.add_argument("--out", default="proton_ke_split.png", help="Output plot filename")
    args = parser.parse_args()

    tree = NuisanceFlatTree(args.f, max_events=args.max_events)

    for id in ['all','qe','2p2h','oth']:
        if id == 'qe':
            mask = _to_numpy(tree.get_mode() == 1) # QE
        elif id == '2p2h':
            mask = _to_numpy(tree.get_mode() == 2) # 2p2h
        elif id == 'oth':
            mask = _to_numpy(tree.get_mode() > 2) # resonant and DIS
        else:
            mask = np.ones(len(tree.get_mode()), dtype=bool) # everything

        tree_copy = NuisanceFlatTree(args.f, max_events=args.max_events) # create a fresh tree copy for each mode to avoid masking issues
        tree_copy.update_tree_with_mask(mask)  

        # Total proton KE per event; None -> 0.0 for empty proton content.
        total_p_ke = ak.fill_none(tree_copy.get_event_variable("total_proton_KE"), 0.0)
        mask_zero = _to_numpy(total_p_ke == 0.0)
        mask_pos = _to_numpy(total_p_ke > 0.0)


        # Muon momentum magnitude from leading muon components.
        mu_px = _to_numpy(tree_copy.get_event_variable("leading_muon_px"))
        mu_py = _to_numpy(tree_copy.get_event_variable("leading_muon_py"))
        mu_pz = _to_numpy(tree_copy.get_event_variable("leading_muon_pz"))
        mu_p = np.sqrt(mu_px ** 2 + mu_py ** 2 + mu_pz ** 2)

        # delta_pt, W, Q2 from flat tree.
        delta_pt = _to_numpy(tree_copy.get_event_variable("dpt"))
        W = _to_numpy(tree_copy.get_event_variable("W"))
        Q2 = _to_numpy(tree_copy.get_event_variable("Q2"))
        mode = _to_numpy(tree_copy.get_mode())

        print(f"{id} - Number of events with total_proton_KE = 0: {np.sum(mask_zero)}/{len(mask_zero)}")
        print(f"{id} - modes with zero recoil: {mode[mask_zero]}")
        print(f"{id} - modes with positive recoil: {mode[mask_pos]}")

        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        axes = axes.ravel()



        plots = [
            (mu_p, "Muon momentum |p_μ| [GeV/c]"),
            (delta_pt, r"δp_T [GeV/c]"),
            (W, "W [GeV]"),
            (Q2, r"Q² [GeV²]"),
            (mode, "Mode")
        ]

        colors = {"zero": "tab:blue", "pos": "tab:orange"}
        labels = {"zero": "total_proton_KE = 0", "pos": "total_proton_KE > 0"}

        for ax, (values, title) in zip(axes, plots):
            bins = 50
            # integer bins for mode
            if title == "Mode":
                bins = np.arange(-0.5, np.max(values) + 1.5, 1)
            ax.hist(values[mask_zero], bins=bins, histtype="step", color=colors["zero"], label=labels["zero"])
            ax.hist(values[mask_pos], bins=bins, histtype="step", color=colors["pos"], label=labels["pos"])
            ax.set_title(title)
            ax.set_ylabel("n events")
            ax.set_xlabel(title)
            ax.set_yscale("log")
            ax.legend()
            ax.grid(alpha=0.3)
        
        

        fig.tight_layout()
        fig_name = args.out.replace(".png", f"_{id}.png")
        fig.savefig(fig_name, dpi=150)
        print(f"Saved {fig_name}")


if __name__ == "__main__":
    main()
