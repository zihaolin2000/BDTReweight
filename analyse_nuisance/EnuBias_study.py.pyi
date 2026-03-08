import sys
# Change this path to your working directory where BDTReweight is installed:
# sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
sys.path.append('/Users/lorenzo/Minerva/fork_reweighting/')

from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



# this script plots the neutrino energy bias (Enu_reco - Enu_true) as a function of the muon transverse momentum and
# the visible hadronic energy (sum of the energy of all protons) in aCCQELike sample

S = 0.028 # GeV, average nucleon separation energy for carbon (can be refined)

# Helper to get arrays and convert masked/None to np.nan
def _get_numeric_array(expr: str, tree: NuisanceFlatTree = None, mask: np.ndarray = None) -> np.ndarray:
    arr = tree.get_event_variable(expr, mask=mask)
    arr = ak.fill_none(arr, np.nan)
    return ak.to_numpy(arr)

def EnuRE(input_file: str,
         pT_bins: np.ndarray = np.linspace(0, 2.0, 21),
         had_bins: np.ndarray = np.linspace(0, 2.0, 21),
         save_fig: str | None = None):
    """
    Compute and plot the mean neutrino energy bias (Enu_reco - Enu_true)
    as a function of muon transverse momentum and visible hadronic energy
    (sum of proton kinetic energies) for CCQELike events.

    Parameters
    ----------
    input_file : str
        Path to the NUISANCE flat tree ROOT file.
    pT_bins : np.ndarray
        Bin edges for muon transverse momentum (GeV/c).
    had_bins : np.ndarray
        Bin edges for visible hadronic energy (GeV).
    save_fig : str | None
        If provided, save the produced figure to this path.

    Returns
    -------
    mean_bias : np.ndarray
        2D array (len(pT_bins)-1, len(had_bins)-1) of mean bias values.
    counts : np.ndarray
        2D array of event counts per bin.
    pT_edges, had_edges : np.ndarray
        The input bin edges (returned for convenience).
    """

    # Load tree
    tree = NuisanceFlatTree(input_file)

    # Selection: CCQELike events
    mask = tree.get_mask_flagCCQELike()



    # Reconstructed and true neutrino energy
    # Use Enu_QE as the commonly provided QE reconstructed energy

    # reco neutrino energy = S + E_mu + sumTp + M_n
    EnuRE = _get_numeric_array('leading_muon_E',tree=tree) + S + _get_numeric_array('total_proton_KE',tree=tree) + 0.939565 # GeV, neutron mass
    Enu_reco = EnuRE
    Enu_true = _get_numeric_array('Enu_true',tree=tree)

    # Muon transverse momentum from leading muon
    mu_px = _get_numeric_array('leading_muon_px',tree=tree)
    mu_py = _get_numeric_array('leading_muon_py',tree=tree)
    pT_mu = np.sqrt(np.nan_to_num(mu_px, nan=0.0)**2 + np.nan_to_num(mu_py, nan=0.0)**2)

    # Visible hadronic energy: sum of proton kinetic energies
    # Use total_proton_KE (returns NaN when no protons) and replace NaN with 0.
    had_ke = _get_numeric_array('total_proton_KE',tree=tree)
    had_ke = np.nan_to_num(had_ke, nan=0.0)

    # Compute bias
    bias = ( Enu_reco - Enu_true ) / Enu_true * 100.0  # percentage bias

    # Remove entries where either reco or true is NaN (invalid)
    valid = np.isfinite(bias) & np.isfinite(pT_mu) & np.isfinite(had_ke)
    if not np.any(valid):
        raise RuntimeError('No valid events found after selection and NaN filtering.')

    pT_vals = pT_mu[valid]
    had_vals = had_ke[valid]
    bias_vals = bias[valid]

    # Compute sum of bias and counts per 2D bin
    sum_bias, pT_edges, had_edges = np.histogram2d(pT_vals, had_vals, bins=[pT_bins, had_bins], weights=bias_vals)
    counts, _, _ = np.histogram2d(pT_vals, had_vals, bins=[pT_bins, had_bins])

    # Compute mean bias per bin; avoid division by zero
    mean_bias = np.full(sum_bias.shape, np.nan)
    nonzero = counts > 0
    mean_bias[nonzero] = sum_bias[nonzero] / counts[nonzero]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    # pcolormesh expects bin edges in X and Y order (x edges -> columns)
    pcm = ax.pcolormesh(pT_edges, had_edges, mean_bias.T, shading='auto', cmap='RdBu_r', vmin=-25, vmax=25)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r'Mean $E_{\nu}^{reco} - E_{\nu}^{true} $(GeV)')
    cbar.set_label(r'(Mean $E_{\nu}^{reco} - E_{\nu}^{true} )/E_{\nu}^{true}$ (%)')
    ax.set_xlabel(r'Muon $p_T$ (GeV/c)')
    ax.set_ylabel(r'$\Sigma T_{p}$ (GeV)')
    ax.set_title('Neutrino energy bias (Enu RE - Enu_true) for CCQELike events')

    # draw a curve at the mean bias = 0 line
    ax.contour(pT_edges[:-1], had_edges[:-1], mean_bias.T, levels=[0], colors='black', linewidths=1)

    if save_fig:
        fig.savefig(save_fig, dpi=150)
        print(f'Saved figure to {save_fig}')
    else:
        plt.show()

    return mean_bias, counts, pT_edges, had_edges

def bias_vs_pT(input_file: str, pT_bins: np.ndarray = np.linspace(0, 2.0, 21), save_fig: str | None = None):
    """
    Compute and plot the mean neutrino energy bias (Enu_reco - Enu_true)
    as a function of muon transverse momentum for CCQELike events.

    Parameters
    ----------
    input_file : str
        Path to the NUISANCE flat tree ROOT file.

    pT_bins : np.ndarray
        Bin edges for muon transverse momentum (GeV/c).
    save_fig : str | None
        If provided, save the produced figure to this path.

    Returns
    -------
    mean_bias : np.ndarray
        1D array of mean bias values per pT bin.
    counts : np.ndarray
        1D array of event counts per pT bin.
    pT_edges : np.ndarray
        The input pT bin edges (returned for convenience).
    """

    # Load tree
    tree = NuisanceFlatTree(input_file)

    EnuRE = _get_numeric_array('leading_muon_E',tree=tree) + S + _get_numeric_array('total_proton_KE',tree=tree)  # GeV, neutron mass
    Enu_reco = EnuRE
    Enu_true = _get_numeric_array('Enu_true',tree=tree)

    # Muon transverse momentum from leading muon
    mu_px = _get_numeric_array('leading_muon_px',tree=tree)
    mu_py = _get_numeric_array('leading_muon_py',tree=tree)
    pT_mu = np.sqrt(np.nan_to_num(mu_px, nan=0.0)**2 + np.nan_to_num(mu_py, nan=0.0)**2)

    # Compute bias
    bias = ( Enu_reco - Enu_true ) / Enu_true * 100.0  # percentage bias

    # Remove entries where either reco or true is NaN (invalid)
    valid = np.isfinite(bias) & np.isfinite(pT_mu)
    if not np.any(valid):
        raise RuntimeError('No valid events found after selection and NaN filtering.')

    pT_vals = pT_mu[valid]
    bias_vals = bias[valid]

    # plot bias (y) vs pT (x) (2D color plot)
    fig, ax = plt.subplots(figsize=(7, 5))
    h = ax.hist2d(pT_vals, bias_vals, bins=[pT_bins,200], cmap='viridis', norm=LogNorm(vmin=1))
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Counts (log scale)')
    ax.set_xlabel(r'Muon $p_T$ (GeV/c)')
    ax.set_ylabel(r'$(E_{\nu}^{reco} - E_{\nu}^{true})/E_{\nu}^{true}$ (%)')
    ax.set_title('Neutrino energy bias vs muon $p_T$ for CCQELike events')
    ax.set_ylim(-25, 25)

    # compute mean bias per pT bin
    sum_bias, pT_edges = np.histogram(pT_vals, bins=pT_bins, weights=bias_vals)
    counts, _ = np.histogram(pT_vals, bins=pT_bins)
    mean_bias = np.full(sum_bias.shape, np.nan)
    nonzero = counts > 0
    mean_bias[nonzero] = sum_bias[nonzero] / counts[nonzero]
    # median bias per pT bin
    median_bias = np.full(sum_bias.shape, np.nan)
    for i in range(len(pT_edges)-1):
        in_bin = (pT_vals >= pT_edges[i]) & (pT_vals < pT_edges[i+1])
        if np.any(in_bin):
            median_bias[i] = np.median(bias_vals[in_bin])

    # plot mean and median bias as a line on top
    pT_bin_centers = 0.5 * (pT_edges[:-1] + pT_edges[1:])
    ax.plot(pT_bin_centers, mean_bias, color='red', marker='.', label='Mean bias')
    ax.plot(pT_bin_centers, median_bias, color='orange', marker='.', label='Median bias')
    ax.legend()

    if save_fig:
        fig.savefig(save_fig, dpi=150)
        print(f'Saved figure to {save_fig}')
    else:
        plt.show()

def bias_vs_had(input_file: str, had_bins: np.ndarray = np.linspace(0, 2.0, 21), save_fig: str | None = None):
    """
    Compute and plot the mean neutrino energy bias (Enu_reco - Enu_true)
    as a function of visible hadronic energy (sum of proton kinetic energies)
    for CCQELike events.

    Parameters
    ----------
    input_file : str
        Path to the NUISANCE flat tree ROOT file.
    had_bins : np.ndarray
        Bin edges for visible hadronic energy (GeV).
    save_fig : str | None
        If provided, save the produced figure to this path.

    Returns
    -------
    mean_bias : np.ndarray
        1D array of mean bias values per hadronic energy bin.
    counts : np.ndarray
        1D array of event counts per hadronic energy bin.
    had_edges : np.ndarray
        The input hadronic energy bin edges (returned for convenience).
    """

    # Load tree
    tree = NuisanceFlatTree(input_file)

    EnuRE = _get_numeric_array('leading_muon_E',tree=tree) + S + _get_numeric_array('total_proton_KE',tree=tree)  # GeV, neutron mass
    Enu_reco = EnuRE
    Enu_true = _get_numeric_array('Enu_true',tree=tree)

    # Visible hadronic energy: sum of proton kinetic energies
    had_ke = _get_numeric_array('total_proton_KE',tree=tree)
    had_ke = np.nan_to_num(had_ke, nan=0.0)

    # Compute bias
    bias = ( Enu_reco - Enu_true ) / Enu_true * 100.0  # percentage bias

    # Remove entries where either reco or true is NaN (invalid)
    valid = np.isfinite(bias) & np.isfinite(had_ke)
    if not np.any(valid):
        raise RuntimeError('No valid events found after selection and NaN filtering.')

    had_vals = had_ke[valid]
    bias_vals = bias[valid]

    # plot bias (y) vs hadronic energy (x) (2D color plot)
    fig, ax = plt.subplots(figsize=(7, 5))
    h = ax.hist2d(had_vals, bias_vals, bins=[had_bins, 200], cmap='viridis', norm=LogNorm(vmin=1))
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Counts (log scale)')
    ax.set_xlabel(r'Visible hadronic energy $\Sigma T_{p}$ (GeV)')
    ax.set_ylabel(r'$(E_{\nu}^{reco} - E_{\nu}^{true})/E_{\nu}^{true}$ (%)')
    ax.set_title('Neutrino energy bias vs visible hadronic energy for CCQELike events')
    ax.set_ylim(-25, 25)

    # compute mean bias per hadronic energy bin
    sum_bias, had_edges = np.histogram(had_vals, bins=had_bins, weights=bias_vals)
    counts, _ = np.histogram(had_vals, bins=had_bins)
    mean_bias = np.full(sum_bias.shape, np.nan)
    nonzero = counts > 0
    mean_bias[nonzero] = sum_bias[nonzero] / counts[nonzero]

    # median bias per hadronic energy bin
    median_bias = np.full(sum_bias.shape, np.nan)
    for i in range(len(had_edges)-1):
        in_bin = (had_vals >= had_edges[i]) & (had_vals < had_edges[i+1])
        if np.any(in_bin):
            median_bias[i] = np.median(bias_vals[in_bin])

    # plot mean and median bias as a line on top
    had_bin_centers = 0.5 * (had_edges[:-1] + had_edges[1:])
    ax.plot(had_bin_centers, mean_bias, color='red', marker='.', label='Mean bias')
    ax.plot(had_bin_centers, median_bias, color='orange', marker='.', label='Median bias')
    ax.legend()

    if save_fig:
        fig.savefig(save_fig, dpi=150)
        print(f'Saved figure to {save_fig}')
    else:
        plt.show()

def bias(input_file: str, pT_bins: np.ndarray = np.linspace(0, 2.0, 21), had_bins: np.ndarray = np.linspace(0, 2.0, 21), save_fig: str | None = None):
    tree = NuisanceFlatTree(input_file)

    EnuRE = _get_numeric_array('leading_muon_E',tree=tree) + S + _get_numeric_array('total_proton_KE',tree=tree)  # GeV, neutron mass
    Enu_reco = EnuRE
    Enu_true = _get_numeric_array('Enu_true',tree=tree)
    print(f"EnuRE sample values (first 10): {EnuRE[:10]}")
    print(f"E_muon sample values (first 10): {_get_numeric_array('leading_muon_E',tree=tree)[:10]}")
    print(f"Total proton KE sample values (first 10): {_get_numeric_array('total_proton_KE',tree=tree)[:10]}")
    print(f"Enu_true sample values (first 10): {_get_numeric_array('Enu_true',tree=tree)[:10]}")

# plot bias (1d histogram)
    bias = ( Enu_reco - Enu_true ) / Enu_true * 100.0  # percentage bias
    plt.hist(bias, bins=100, range=(-100, 100), color='blue', alpha=0.7)
    plt.xlabel(r'$(E_{\nu}^{reco} - E_{\nu}^{true})/E_{\nu}^{true}$ (%)')
    plt.ylabel('Counts')
    plt.title('Neutrino energy bias distribution for CCQELike events')
    plt.xlim(-100, 100)
    if save_fig:
        plt.savefig(save_fig, dpi=150)
        print(f'Saved figure to {save_fig}')
    plt.show()


if __name__ == "__main__":
    # Example usage
    input_file = "/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"
    tree_name = "FlatTree_VARS"

    # Load the nuisance flat tree (the class opens FlatTree_VARS internally)
    nuisance_tree = NuisanceFlatTree(input_file)


    # Access a specific branch (e.g., "Enu_true") via get_event_variable
    enu_values = nuisance_tree.get_event_variable("Enu_true")
    # Convert awkward array to numpy for simple inspect
    enu_values = ak.fill_none(enu_values, np.nan)
    enu_values = ak.to_numpy(enu_values)
    print(f"First 10 Enu_true values: {enu_values[:10]}")

    # Run the study and show the plot
    try:
        # bias(input_file, save_fig=None,pT_bins=np.linspace(0, 2.5, 100),had_bins=np.linspace(0, 1.4, 100))
        bias_vs_had(input_file, had_bins=np.linspace(0, 1.4, 100), save_fig="bias_vs_had.png")

        bias_vs_pT(input_file, pT_bins=np.linspace(0, 2.5, 100), save_fig="bias_vs_pT.png")


        EnuRE(input_file, save_fig="bias_vs_pTSumTp.png",pT_bins=np.linspace(0, 2.5, 40),had_bins=np.linspace(0, 1.4, 40))
    except Exception as e:
        print(f"EnuRE failed: {e}")
