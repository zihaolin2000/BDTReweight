import sys
# Change this path to your working directory where BDTReweight is installed:
# sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
sys.path.append('/Users/lorenzo/Minerva/fork_reweighting/')

from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import ROOT


def plot_xsec_vs_enu_by_reaction(nuisance_tree, outname, bins=50, flux_histogram=None):
    """Plot cross-section-weighted event counts vs true neutrino energy for several reaction categories.

    The histograms are computed as weighted counts per energy bin and plotted as lines connecting
    bin centers (instead of bars).
    """
    # Get the true neutrino energy and the cross-section from the nuisance tree
    enu_true = nuisance_tree.get_event_variable("Enu_true")
    xsec = nuisance_tree.get_event_variable("fScaleFactor")

    # Convert awkward arrays to numpy for plotting
    enu_true = ak.fill_none(enu_true, np.nan)
    xsec = ak.fill_none(xsec, np.nan)
    enu_true = ak.to_numpy(enu_true)
    xsec = ak.to_numpy(xsec)

    # Safely get Mode as a numpy array (fill missing with -1)
    mode_arr = ak.fill_none(nuisance_tree.get_event_variable("Mode"), -1)
    mode_arr = ak.to_numpy(mode_arr)

    # Build boolean masks for reactions
    mask_qe = mode_arr == 1
    mask_res = np.isin(mode_arr, [11, 12, 13, 17, 22, 23])
    mask_2p2h = mode_arr == 2
    mask_dis = np.isin(mode_arr, [21, 26])

    # Calculate fractions for legend labels
    total_events = len(mode_arr)
    qe_frac = np.sum(mask_qe) / total_events if total_events > 0 else 0
    res_frac = np.sum(mask_res) / total_events if total_events > 0 else 0
    p2h_frac = np.sum(mask_2p2h) / total_events if total_events > 0 else 0
    dis_frac = np.sum(mask_dis) / total_events if total_events > 0 else 0

    # Make sure we have finite entries to determine binning
    finite_mask = np.isfinite(enu_true) & np.isfinite(xsec)
    if not np.any(finite_mask):
        print("No finite Enu_true/xsec entries found - nothing to plot.")
        return
    valid_enu = enu_true[finite_mask]

    emin = 0
    emax = 20
    if np.isclose(emin, emax):
        # Expand a tiny bit so linspace can make bins
        emin = emin - 0.5
        emax = emax + 0.5

    # Create a common set of bin edges
    bin_edges = np.linspace(emin, emax, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(10, 6))

    # For each reaction, compute weighted histogram and plot as a line connecting bin centers
    def plot_reaction(mask, label, color):
        sel = mask & np.isfinite(enu_true) & np.isfinite(xsec)
        if np.count_nonzero(sel) == 0:
            # nothing to plot
            return
        counts, _ = np.histogram(enu_true[sel], bins=bin_edges, weights=xsec[sel]/bin_width)
        plt.plot(bin_centers, counts, label=label, color=color, linewidth=2)

    plot_reaction(mask_qe, f'QE ({qe_frac*100:.1f}%)', 'blue')
    plot_reaction(mask_res, f'RES ({res_frac*100:.1f}%)', 'orange')
    plot_reaction(mask_2p2h, f'2p2h ({p2h_frac*100:.1f}%)', 'green')
    plot_reaction(mask_dis, f'DIS+multi-$\pi$  ({dis_frac*100:.1f}%)', 'red')

    plt.xlabel(r'$E_\nu$ (GeV)')
    plt.ylabel(r'$\frac{d\sigma}{dE_\nu} (cm^2/GeV)$')


    # overlay the flux histogram (its a TH1D)
    if flux_histogram is not None:
        flux_bin_edges = flux_histogram.GetXaxis().GetXbins()
        flux_bin_edges = np.array([flux_bin_edges.At(i) for i in range(flux_bin_edges.GetSize())])
        flux_bin_centers = 0.5 * (flux_bin_edges[:-1] + flux_bin_edges[1:])
        flux_values = np.array([flux_histogram.GetBinContent(i) for i in range(1, flux_histogram.GetNbinsX() + 1)])
        # rescale so that the max is 0.9* the max of the xsec curves (for better visibility)
        max_xsec = max([plt.gca().get_lines()[i].get_ydata().max() for i in range(plt.gca().get_lines().__len__())])
        flux_values = flux_values / flux_values.max() * 0.9 * max_xsec

        plt.hist(flux_bin_centers, bins=flux_bin_edges, weights=flux_values, label='Flux', color='gray', alpha=0.5, histtype='stepfilled')

    # crop x-axis between 0 and 20 GeV
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    plt.savefig(outname, dpi=300)
    # plt.show()



if __name__ == "__main__":
    # Example usage
    input_file = "/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT.root"
    tree_name = "FlatTree_VARS"

    f = ROOT.TFile.Open(input_file)
    print(f"Opened file: {input_file}")
    print("So, what's in this file?")
    for key in f.GetListOfKeys():
        print(key.GetName(), key.GetClassName())
    flux_histo_name = "FlatTree_FLUX"
    h_flux = f.Get(flux_histo_name)
    h_flux.SetDirectory(0)  # Detach histogram from file so it won't be deleted when file is closed
    f.Close()

    # Load the nuisance flat tree (the class opens FlatTree_VARS internally)
    nuisance_tree = NuisanceFlatTree(input_file)


    # Access a specific branch (e.g., "Enu_true") via get_event_variable
    enu_values = nuisance_tree.get_event_variable("Enu_true")
    # Convert awkward array to numpy for simple inspect
    enu_values = ak.fill_none(enu_values, np.nan)
    enu_values = ak.to_numpy(enu_values)
    print(f"First 10 Enu_true values: {enu_values[:10]}")

    # Plot cross-section vs neutrino energy
    plot_xsec_vs_enu_by_reaction(nuisance_tree, bins=40, flux_histogram=h_flux, outname="xsec_vs_enu_by_reaction_NEUT.png")
    # repeat for CCQELike only (signal)
    input_file_ccqelike = input_file.replace(".root", "_CCQELike.root")
    nuisance_tree_ccqelike = NuisanceFlatTree(input_file_ccqelike)
    plot_xsec_vs_enu_by_reaction(nuisance_tree_ccqelike, bins=40, outname="xsec_vs_enu_by_reaction_NEUT_ccqelike.png")
