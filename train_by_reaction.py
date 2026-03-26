import os
import sys
# Change this path to your working directory where BDTReweight is installed:
# sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
sys.path.append('/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/')

from BDTReweight.analysis import transform_momentum_to_reaction_frame, create_dataframe_from_nuisance, draw_source_target_distributions_and_ratio
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
from BDTReweight.utilities import particle_variable_to_latex, diff_xsec_latex_wrt_variable
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import pathlib
import re
import joblib
import ROOT
import pickle
import argparse


MUON_MASS_GEV = 0.1056583745
NUCLEON_MASS_GEV = 0.939565
S_RE_GEV = 0.028
K_F_GEV = 0.228
E_SHIFT_GEV = 0.020

MUON_PT_BIN_EDGES_GEV = np.array([
    0.0, 0.075, 0.15, 0.25, 0.325, 0.4, 0.475, 0.55,
    0.7, 0.85, 1.0, 1.25, 1.75, 2.5
], dtype=float)

RECOIL_BIN_EDGES_MEV = np.array([
    0.0, 20.0, 40.0, 80.0, 120.0, 160.0,
    240.0, 320.0, 400.0, 600.0, 800.0, 1400.0
], dtype=float)

PSI_PRIME_BIN_EDGES = np.array([
    -10.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5,
    -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
], dtype=float)

# Default per-topology configuration; values copied from the 0p0n defaults below.
CATEGORY_CONFIGS = {
    '0p0n': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime', 'weight'],
    },
    '0pNn': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '>=1'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz', 'leading_neutron_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': [
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz','psi_prime'
        ],
        'particle_names': ['leading_neutron','total_proton'],
        'drawing_variables': [
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 'weight'
        ],
    },
    '1p0n': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    '1pNn': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    '2p0n': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    '2pNn': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    'others': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
}


def compute_psi_prime(q0, q3_mag, k_f=K_F_GEV, e_shift=E_SHIFT_GEV):
    q0 = np.asarray(q0, dtype=float)
    q3_mag = np.asarray(q3_mag, dtype=float)

    eta_f = k_f / NUCLEON_MASS_GEV
    kappa = q3_mag / (2.0 * NUCLEON_MASS_GEV)
    lambda_var = (q0 - e_shift) / (2.0 * NUCLEON_MASS_GEV)
    tau = kappa * kappa - lambda_var * lambda_var

    normalizing_inner = np.sqrt(1.0 + eta_f * eta_f) - 1.0
    if normalizing_inner <= 0.0:
        return np.full_like(q0, np.nan, dtype=float)
    normalizing_factor = 1.0 / np.sqrt(normalizing_inner)

    tau_term = tau + tau * tau
    sqrt_tau_term = np.sqrt(np.clip(tau_term, 0.0, None))
    denominator_sq = (1.0 + lambda_var) * tau + kappa * sqrt_tau_term

    valid = (tau_term >= 0.0) & (denominator_sq > 0.0)
    psi_prime = np.full_like(q0, np.nan, dtype=float)
    denominator = np.sqrt(np.clip(denominator_sq, 0.0, None))
    psi_prime[valid] = ((lambda_var - tau) / denominator * normalizing_factor)[valid]
    return psi_prime


def get_psi_prime_from_fs_kinematics(recoil_gev, muon_px_beam, muon_py_beam, muon_pz_beam):
    recoil_gev = np.asarray(recoil_gev, dtype=float)
    muon_px_beam = np.asarray(muon_px_beam, dtype=float)
    muon_py_beam = np.asarray(muon_py_beam, dtype=float)
    muon_pz_beam = np.asarray(muon_pz_beam, dtype=float)

    muon_e = np.sqrt(
        muon_px_beam * muon_px_beam
        + muon_py_beam * muon_py_beam
        + muon_pz_beam * muon_pz_beam
        + MUON_MASS_GEV * MUON_MASS_GEV
    )
    q0 = recoil_gev + S_RE_GEV
    qx = -muon_px_beam
    qy = -muon_py_beam
    q3 = muon_e - muon_pz_beam + recoil_gev + S_RE_GEV
    q_mag = np.sqrt(qx * qx + qy * qy + q3 * q3)

    return compute_psi_prime(q0, q_mag)


def _format_bin_edge(value):
    return f"{value:g}".replace('.', 'p')


def _hist_density_mean(values, weights, bin_edges):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return np.nan
    values = values[valid]
    weights = weights[valid]

    counts, edges = np.histogram(values, bins=np.asarray(bin_edges, dtype=float), weights=weights)
    bin_widths = np.diff(edges)
    updated_bin_content = counts / bin_widths
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    norm = np.sum(updated_bin_content)
    if norm <= 0.0:
        return np.nan
    return np.sum(updated_bin_content * bin_centers) / norm


def save_mean_vs_slice_plot(
    x_centers,
    source_means,
    target_means,
    reweighted_means,
    x_label,
    slice_name,
    unit,
    process,
    category,
    output_dir,
):
    fig, (ax_main, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        dpi=200,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
        sharex=True,
    )

    source_means = np.asarray(source_means, dtype=float)
    target_means = np.asarray(target_means, dtype=float)
    reweighted_means = np.asarray(reweighted_means, dtype=float)

    ax_main.plot(x_centers, source_means, 'o-', label='Source', color='tab:green')
    ax_main.plot(x_centers, target_means, 'o-', label='Target', color='tab:red')
    ax_main.plot(x_centers, reweighted_means, 'o-', label='Source (Reweighted)', color='tab:blue')
    ax_main.set_ylabel(r'Mean $\psi^\prime$')
    ax_main.legend(loc='best')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(
        f"Mean $\\psi^\\prime$ vs {slice_name} ({unit}). Process: {process}, category: {category}",
        fontsize=12,
    )

    diff_target_source = target_means - source_means
    diff_reweighted_source = reweighted_means - source_means
    ax_diff.plot(x_centers, diff_target_source, 'o-', color='tab:orange', label='Target - Source')
    ax_diff.plot(
        x_centers,
        diff_reweighted_source,
        'o-',
        color='tab:purple',
        label='Reweighted - Source',
    )
    ax_diff.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax_diff.set_xlabel(f'{x_label} [{unit}]')
    ax_diff.set_ylabel(r'$\Delta$ mean')
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(loc='best', fontsize=8)

    output_name = f"mean_vs_{slice_name}_{process}_{category}.png"
    fig.savefig(f"{output_dir}{output_name}", bbox_inches='tight')
    print(f"Saved mean-vs-{slice_name} figure to {output_name}")
    plt.close(fig)


def save_psi_prime_slice_plot(
    source_df,
    target_df,
    source_weights,
    target_weights,
    new_source_weights,
    source_mask,
    target_mask,
    pics_folder_name,
    process,
    category,
    slice_type,
    bin_index,
    low,
    high,
    unit,
):
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    n_source = int(np.sum(source_mask))
    n_target = int(np.sum(target_mask))

    if n_source == 0 or n_target == 0:
        print(
            f"Skipping {slice_type} slice [{low:g}, {high:g}] {unit}: "
            f"source events={n_source}, target events={n_target}"
        )
        return

    source_slice = source_df.iloc[source_mask]
    target_slice = target_df.iloc[target_mask]
    source_weights_slice = np.asarray(source_weights, dtype=float)[source_mask]
    target_weights_slice = np.asarray(target_weights, dtype=float)[target_mask]
    new_source_weights_slice = np.asarray(new_source_weights, dtype=float)[source_mask]

    fig = draw_source_target_distributions_and_ratio(
        source_slice,
        target_slice,
        variables=['psi_prime'],
        source_weights=source_weights_slice,
        target_weights=target_weights_slice,
        new_source_weights=new_source_weights_slice,
        legends=['Source', 'Source (Reweighted)', 'Target'],
        variable_bins={'psi_prime': PSI_PRIME_BIN_EDGES},
    )

    fig.suptitle(
        f"Psi-prime. {slice_type} bin {bin_index}: [{low:g}, {high:g}] {unit}. "
        f"Process: {process}, category: {category}",
        fontsize=16,
    )
    output_name = f"PsiPrime_{slice_type}Slice_bin{bin_index}_{process}_{category}.png"
    fig.savefig(f"{pics_folder_name}{output_name}")
    print(f"Saved psi-prime slice plot to {output_name}")
    plt.close()


# HERE STARTS THE MAIN FUNCTION


# arguments parser
p = argparse.ArgumentParser(description='Train BDT reweighter by reaction channel.')
p.add_argument('--source_path', '-s', type=str, help='Path to the source model ROOT file.')
p.add_argument('--target_path', '-t', type=str, help='Path to the target model ROOT file.')
p.add_argument('--module_path', '-m', type=str, help='Path to the BDTReweight module.')
p.add_argument('--model_name', type=str, help='Identifier of the target model.')
p.add_argument('--build_tree_of_weights',action='store_true', help='Activate building a ROOT TTree with the reweighting weights.')
p.add_argument('--shape_only', action='store_true', help='Only reweight shape, do not change total cross section')
p.add_argument('--max_events', type=int, default=None, help='Maximum number of events to use for training (for both source and target).')
p.add_argument('--plots_dir', type=str, default=None, help='Full output directory for plots. If set, this path is used directly.')
p.add_argument('--category', type=str, default='0p0n', help='Reaction category to train on (e.g. 0p0n, 1p0n, etc.).')

build_tree_of_weights = False

# get build_tree_of_weights from command line arguments
args = p.parse_args()
if args.build_tree_of_weights:
    build_tree_of_weights = True

target_path = args.target_path
source_path = args.source_path
if args.module_path:
    sys.path.append(args.module_path)

# target_path = '/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root'
# target_path = '/eos/user/l/lgiannes/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root'
# source_path = '/Users/lorenzo/cernbox/MINERVA_MC/source/ReweightSourceCCQELike_minervame1L.root'
# source_path = '/Users/lorenzo/cernbox/MINERVA_MC/source/minervame1L_for_rwg.root'
# source_path = '/eos/user/l/lgiannes/MINERVA_MC/source/minervame1L_for_rwg.root'
# source_path = '/Users/lorenzo/cernbox/MINERVA_MC/source/ReweightSourceCCQELike_minervame1M.root'

if args.model_name:
    target_model_name = args.model_name
else:
    target_model_name = pathlib.Path(target_path).stem
    target_model_name = re.search(r'MINERvAflux_([^_]+)_', target_model_name).group(1)
    if target_model_name is None:
        print("CAN'T IDENTIFY TARGET MODEL NAME! ABORT!")
        exit()

print(f'Reweighting to target model: {target_model_name}')



tree_source_train = uproot.open(source_path)['EventKinematics_truth'].arrays(library='pd')
if args.max_events is not None:
    print(f"Limiting number of events to {args.max_events} for source model.")
    tree_source_train = tree_source_train.iloc[:args.max_events]

topologies = {0:'0p0n',1:'0pNn',2:'1p0n',3:'1pNn',4:'2p0n',5:'2pNn',6:'others'}
tree_source_train['topology'] = tree_source_train['topology'].map(topologies)
tree_source_train = tree_source_train.rename(columns={'muon_px':'leading_muon_px', 'muon_py':'leading_muon_py', 'muon_pz':'leading_muon_pz',
    'sum_p_px':'total_proton_px', 'sum_p_py':'total_proton_py', 'sum_p_pz':'total_proton_pz', 'sum_Tp':'total_proton_KE', 'leading_n_px':'leading_neutron_px',
    'leading_n_py':'leading_neutron_py', 'leading_n_pz':'leading_neutron_pz', 'leading_p_px':'leading_proton_px', 'leading_p_py':'leading_proton_py',
    'leading_p_pz':'leading_proton_pz', 'subleading_p_px':'subleading_proton_px', 'subleading_p_py':'subleading_proton_py', 'subleading_p_pz':'subleading_proton_pz'}
)

plt.figure()
plt.hist(tree_source_train[tree_source_train['topology']=='0p0n']['total_proton_KE'], bins = 300, label='source model',alpha=0.5, range=(0.001,2.), weights=tree_source_train[tree_source_train['topology']=='0p0n']['init_wgt'])
plt.xlabel(r'$\sum T_{p}$ [GeV]')
# y in log scale
# plt.yscale('log')
plt.ylabel('counts')

if args.plots_dir is not None:
    plot_root = pathlib.Path(args.plots_dir).expanduser().resolve()
else:
    plot_root = pathlib.Path(args.module_path) / "pics" / target_model_name

plot_root.mkdir(parents=True, exist_ok=True)
pics_folder_name = str(plot_root) + "/"

plt.savefig(f'{pics_folder_name}sum_Tp_source_model_0p0n.png')
print("Saved sum_Tp_source_model_0p0n.png")
plt.close()

# print(tree_source_train.keys())

source_train = {}
source_test = {}
source_total = {}
for topology in topologies.values():
    source_train[topology] = tree_source_train[tree_source_train['topology']==topology].copy()
    # create a temporary test set 
    source_test[topology] = source_train[topology].iloc[np.arange(0, int(len(source_train[topology])/7.53),1)].copy()
    source_total[topology] = source_train[topology]

# Load the target tree to compute the total cross section.
tree_target_train = NuisanceFlatTree(target_path)
# if 'neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_' in target_path:
    # target_is_from_hadded = True
target_is_from_hadded = False
# this is a bit silly: since I did hadd on nuisance flat trees, the total xsec is multiplied by the number of files I hadded (10)
target_ccqelike_xsec = tree_target_train.get_total_xsec()
if target_is_from_hadded:
    target_ccqelike_xsec /= 10 # divide by the number of files I hadded to get the correct total xsec for the target model (weird NUISANCE behavior...)
if args.max_events is not None:
    print(f"Limiting number of events to {args.max_events} for both source and target.")
    # cut all numpy arrays in the NuisanceFlatTree to max_events
    tree_target_train = NuisanceFlatTree(target_path, max_events=args.max_events)


target_train = {}
target_test = {}
# Specify detecting thresholds and topology particle counts:
KE_thresholds={'proton':50, 'neutron':10} # (MeV) use very large thresholds if you want to effectively put everything in 0p0n samples
# scale_source_train = len(tree_target_train._flattree_vars)/len(tree_source_train)
scale_source_train = 1 # 2.489225788674492e-44
# The following factor is used to set the total xsec.
# It should be the ratio between the total xsec predicted by the target model over that predicted by the source model. (σ_target / σ_source)
scale_target_train = 1 # 1.84e-43

# Quick overview plot: event counts per category for source and target before any further processing.
category_order = list(CATEGORY_CONFIGS.keys())
source_counts = [int(np.sum(tree_source_train['topology'] == cat)) for cat in category_order]
target_counts = []
for cat in category_order:
    pc = CATEGORY_CONFIGS[cat]['particle_counts']
    mask = np.asarray(tree_target_train.get_mask_topology(particle_counts=pc, KE_thresholds=KE_thresholds), dtype=bool)
    target_counts.append(int(np.sum(mask)))

x = np.arange(len(category_order))
width = 0.4
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.bar(x - width/2, source_counts, width, label='Source', color='tab:green', alpha=0.7)
ax.bar(x + width/2, target_counts, width, label='Target', color='tab:blue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(category_order, rotation=30, ha='right')
ax.set_ylabel('Events')
ax.set_title('Event counts per category (pre-selection)')
ax.legend()
ax.grid(axis='y', alpha=0.2)

for i, count in enumerate(source_counts):
    ax.text(x[i] - width/2, count, f"{count}", ha='center', va='bottom', fontsize=8, color='tab:green')
for i, count in enumerate(target_counts):
    ax.text(x[i] + width/2, count, f"{count}", ha='center', va='bottom', fontsize=8, color='tab:blue')

category_plot_name = f"{pics_folder_name}category_counts_source_target.png"
fig.tight_layout()
fig.savefig(category_plot_name)
print(f"Saved category count plot to {category_plot_name}")
plt.close(fig)

# Drop target events with zero proton kinetic energy to avoid unphysical entries in training.
# GENIEv3 has a bug with events with zero proton KE. Remove them
if (target_model_name == 'GENIEv3'):
    target_rows_before = tree_target_train.get_n_entries()
    positive_recoil_mask = np.asarray(tree_target_train.get_mask_positive_recoil_energy(), dtype=bool)
    tree_target_train.update_tree_with_mask(positive_recoil_mask)
    removed_zero_ke = target_rows_before - tree_target_train.get_n_entries()
    print(f"Because this is model GENIEv3, we remove events with zero proton kinetic energy cause they look ill-defined in the target flat tree.")
    print(f"Removed {removed_zero_ke} target events with zero proton kinetic energy")

# extract cross section from source model file
source_file = ROOT.TFile(source_path)
h_xsec_ccqelike = ROOT.TH1D(source_file.Get('h_eventRate_qelike_cross_section'))
source_ccqelike_xsec = h_xsec_ccqelike.GetBinContent(1)
h_xsec_total = ROOT.TH1D(source_file.Get('h_eventRate_mc_cross_section'))
source_total_xsec = h_xsec_total.GetBinContent(1)
# h_xsec_ccqelike_qe = source_file['h_eventRate_qelike_qe_cross_section']
# h_xsec_tot = source_file['h_eventRate_mc_cross_section']
# xsec is just the bin content of the histogram (only one bin)
print(f"Total xsec from source model: {source_total_xsec*1e38:.2f} x 10^-38 cm^2")
print(f"Total CCQELike xsec from source model: {source_ccqelike_xsec*1e38:.2f} x 10^-38 cm^2")
print(f"Total CCQELike xsec from target model: {target_ccqelike_xsec*1e38:.2f} x 10^-38 cm^2")

scale_target_train = target_ccqelike_xsec / source_ccqelike_xsec


# # Test setup: keep all QE events, plus exactly one 2p2h (Mode==2) and one Oth (Mode>2) if they exist.
# mode_arr = tree_target_train.get_mode()
# qe_mask = mode_arr == 1
# two_p2h_indices = np.where(mode_arr == 2)[0]
# oth_indices = np.where(mode_arr > 2)[0]

# keep_indices = list(np.where(qe_mask)[0])
# if len(two_p2h_indices) > 0:
#     keep_indices.append(int(two_p2h_indices[0]))
# if len(oth_indices) > 0:
#     keep_indices.append(int(oth_indices[0]))

# keep_indices = np.array(sorted(set(keep_indices)), dtype=int)
# mask_keep = np.full(len(mode_arr), False)
# mask_keep[keep_indices] = True

# tree_target_train.update_tree_with_mask(mask_keep)
# tree_target_train._total_xsec = np.sum(tree_target_train._flattree_vars['fScaleFactor'])
# print(
#     f"QE+1(2p2h)+1(Oth) test active: kept {len(keep_indices)} events (QE={np.sum(qe_mask)}, "
#     f"2p2h_kept={len(two_p2h_indices)>0}, Oth_kept={len(oth_indices)>0})"
# )


if args.shape_only:
    print('Ignoring total cross section and modifying only shape')
    scale_target_train = 1.0

# Category name:
category = args.category

if category not in CATEGORY_CONFIGS:
    raise ValueError(f"Unknown category '{category}'. Available: {list(CATEGORY_CONFIGS.keys())}")

particle_counts = CATEGORY_CONFIGS[category]['particle_counts']
variable_exprs = CATEGORY_CONFIGS[category]['variable_exprs']
reweight_variables = CATEGORY_CONFIGS[category]['reweight_variables']
particle_names = CATEGORY_CONFIGS[category]['particle_names']
drawing_variables = CATEGORY_CONFIGS[category]['drawing_variables']

source_total = len(source_train[category])
print("Number of events:")
print(f"SOURCE: True QE events:      {np.sum(source_train[category]['reactionCode']==1)} ({np.sum(source_train[category]['reactionCode']==1)/source_total*100:.2f} %)")
print(f"SOURCE: True 2p2h events:    {np.sum(source_train[category]['reactionCode']==2)} ({np.sum(source_train[category]['reactionCode']==2)/source_total*100:.2f} %)")
print(f"SOURCE: True RES+DIS events: {np.sum(source_train[category]['reactionCode']>2)} ({np.sum(source_train[category]['reactionCode']>2)/source_total*100:.2f} %)")
target_total = len(tree_target_train._flattree_vars)
print(f"TARGET: True QE events:      {np.sum(tree_target_train.get_mode()==1)} ({np.sum(tree_target_train.get_mode()==1)/target_total*100:.2f} %)")
print(f"TARGET: True 2p2h events:    {np.sum(tree_target_train.get_mode()==2)} ({np.sum(tree_target_train.get_mode()==2)/target_total*100:.2f} %)")
print(f"TARGET: True RES+DIS events: {np.sum(tree_target_train.get_mode()>2)} ({np.sum(tree_target_train.get_mode()>2)/target_total*100:.2f} %)")

scale_target_train *= float(source_total / target_total)

print("Event rates:")
source_total_event_rate = scale_source_train * np.sum(source_train[category]['init_wgt'])
source_qe_event_rate = scale_source_train * np.sum(source_train[category]['init_wgt'][source_train[category]['reactionCode']==1])
source_2p2h_event_rate = scale_source_train * np.sum(source_train[category]['init_wgt'][source_train[category]['reactionCode']==2])
source_resdis_event_rate = scale_source_train * np.sum(source_train[category]['init_wgt'][source_train[category]['reactionCode']>2])
target_total_event_rate = scale_target_train * len(tree_target_train._flattree_vars)
target_qe_event_rate = scale_target_train * np.sum(tree_target_train.get_mode()==1)
target_2p2h_event_rate = scale_target_train * np.sum(tree_target_train.get_mode()==2)
target_resdis_event_rate = scale_target_train * np.sum(tree_target_train.get_mode()>2)
print(f"SOURCE QE event rate:      {source_qe_event_rate:.0f} ({source_qe_event_rate/source_total_event_rate*100:.2f} % )")
print(f"SOURCE 2p2h event rate:    {source_2p2h_event_rate:.0f} ({source_2p2h_event_rate/source_total_event_rate*100:.2f} % )")
print(f"SOURCE RES+DIS event rate: {source_resdis_event_rate:.0f} ({source_resdis_event_rate/source_total_event_rate*100:.2f} % )")
print(f"TARGET QE event rate:      {target_qe_event_rate:.0f} ({target_qe_event_rate/target_total_event_rate*100:.2f} % )")
print(f"TARGET 2p2h event rate:    {target_2p2h_event_rate:.0f} ({target_2p2h_event_rate/target_total_event_rate*100:.2f} % )")
print(f"TARGET RES+DIS event rate: {target_resdis_event_rate:.0f} ({target_resdis_event_rate/target_total_event_rate*100:.2f} % )")


print(f"Training on variables: {', '.join(reweight_variables)}")

dict_to_tree = {}
all_source_plot_chunks = []
all_target_plot_chunks = []

for process in ['Oth','2p2h','QE']:
    process_pics_folder = f'{pics_folder_name}{process}/'
    os.makedirs(process_pics_folder, exist_ok=True)

    target_mask = np.asarray(tree_target_train.get_mask_topology(particle_counts = particle_counts, KE_thresholds = KE_thresholds), dtype=bool)
    # source_mask = np.ones(len(source_train[category]), dtype=bool)
    if process == 'QE':
        print("\nReweighting process: QE")
        source_mask = source_train[category]['reactionCode'] == 1
        target_mask &= (tree_target_train.get_mode() == 1)
    elif process == '2p2h':
        print("\nReweighting process: 2p2h")
        source_mask = source_train[category]['reactionCode'] == 2
        target_mask &= (tree_target_train.get_mode() == 2)
    elif process == 'Oth':
        print("\nReweighting process: Other")
        source_mask = source_train[category]['reactionCode'] > 2
        target_mask &= (tree_target_train.get_mode() > 2)
    else:
        raise ValueError(f"Unknown process: {process}")

    target_train[category] = create_dataframe_from_nuisance(tree_target_train, variable_exprs=variable_exprs, mask=target_mask)
    target_train[category] = transform_momentum_to_reaction_frame(target_train[category], selector_lepton='leading_muon', particle_names=particle_names)
    target_train[category]['weight'] = scale_target_train

    # check for negative total_proton_KE in target_train and print how many events have it, then drop those events
    n_negative_ke = np.sum(target_train[category]['total_proton_KE'] < 0)
    if n_negative_ke > 0:
        print(f"Warning: found {n_negative_ke} events with negative total_proton_KE in target_train for category {category}. These events will be dropped.")
        target_train[category] = target_train[category][target_train[category]['total_proton_KE'] >= 0]

    source_train_p = source_train[category][source_mask].copy()
    target_train_p = target_train[category].copy()



    # Build derived variables for training and diagnostics (psi_prime depends on recoil and muon kinematics).
    source_muon_py = source_train_p['leading_muon_py'].to_numpy()
    source_muon_pz = source_train_p['leading_muon_pz'].to_numpy()
    target_muon_py = target_train_p['leading_muon_py'].to_numpy()
    target_muon_pz = target_train_p['leading_muon_pz'].to_numpy()
    source_muon_px = np.zeros_like(source_muon_py)
    target_muon_px = np.zeros_like(target_muon_py)

    source_train_p['muon_pt_gev'] = np.abs(source_muon_py)
    target_train_p['muon_pt_gev'] = np.abs(target_muon_py)

    source_train_p['recoil_gev'] = np.nan_to_num(source_train_p['total_proton_KE'].to_numpy(), nan=0.0)
    target_train_p['recoil_gev'] = np.nan_to_num(target_train_p['total_proton_KE'].to_numpy(), nan=0.0)
    source_train_p['recoil_mev'] = 1000.0 * source_train_p['recoil_gev']
    target_train_p['recoil_mev'] = 1000.0 * target_train_p['recoil_gev']

    source_train_p['psi_prime'] = get_psi_prime_from_fs_kinematics(
        recoil_gev=source_train_p['recoil_gev'].to_numpy(),
        muon_px_beam=source_muon_px,
        muon_py_beam=source_muon_py,
        muon_pz_beam=source_muon_pz,
    )
    target_train_p['psi_prime'] = get_psi_prime_from_fs_kinematics(
        recoil_gev=target_train_p['recoil_gev'].to_numpy(),
        muon_px_beam=target_muon_px,
        muon_py_beam=target_muon_py,
        muon_pz_beam=target_muon_pz,
    )

    # Create test samples after derived columns are present so they include psi_prime.
    source_test_p = source_train_p.iloc[np.arange(0, int(len(source_train_p)/10),1)].copy()
    target_test_p = target_train_p.copy()

    print(f"Source sample shape: {source_train_p[reweight_variables].shape}")
    print(f"Target sample shape: {target_train_p[reweight_variables].shape}")

    print("Fitting reweighter...")
    reweighter = Reweighter(n_estimators=100, learning_rate=0.4, max_depth=4, min_samples_leaf=30, gb_args={'subsample': 1.0})
    reweighter.fit(original=source_train_p[reweight_variables], target=target_train_p[reweight_variables],
                   # target_weight=target_train_p['weight'],
                   # original_weight=None
                   )

    print("Saving model ...", end='')
    gb_model = getattr(reweighter, '_gb', getattr(reweighter, 'gb'))
    output_model_path = pathlib.Path(target_path).parent
    output_model_path = output_model_path / 'BDTreweight_outputs'
    output_model_path.mkdir(parents=True, exist_ok=True)

    # joblib.dump(gb_model, output_model_path / f'GBReweighterModel_{target_model_name}_{process}_{category}.pkl')
    pickle_output_file = output_model_path / target_model_name / process / f'GBReweighterModel_{category}.pkl'
    os.makedirs(pickle_output_file.parent, exist_ok=True)
    # force protocol to be readable by python 3.9
    pickle.dump(reweighter, open(pickle_output_file, 'wb'), protocol=4)
    print(f" Done. Pickle saved to {pickle_output_file}")

    test_weights = reweighter.predict_matched_total_weights(
        source_test_p[reweight_variables],
        # original_weight=None,
        target_weight=target_test_p['weight']
    )
    all_weights = reweighter.predict_matched_total_weights(
        source_train_p[reweight_variables],
        # original_weight=None,
        target_weight=target_train_p['weight']
    )

    target_n_events = np.sum(target_test_p['weight'])
    source_n_events_before = np.sum(source_train_p['init_wgt'])
    source_n_events_after = np.sum(all_weights)
    print(f"Target n. events: {target_n_events}")
    # print(f"Source n. events before reweighting: {source_n_events_before}")
    # print(f"Source n. events after reweighting: {source_n_events_after}")

    fig = draw_source_target_distributions_and_ratio(source_train_p, target_train_p,
        variables = drawing_variables,
        source_weights = source_train_p['init_wgt'],
        target_weights = target_train_p['weight'],
        new_source_weights = all_weights,
        legends = ['Source', 'Source (Reweighted)', 'Target'],
        # xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
        # ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
        # scale_target = scale_target_train
    )

    # add gloabal title to the figure
    fig.suptitle(f'Reweighting Result for process: {process} in category: {category}', fontsize=16)
    fig.savefig(f'{process_pics_folder}ReweightingResult_{process}_{category}.png')
    print(f"Saved reweighting result figure to ReweightingResult_{process}_{category}.png")
    plt.close()

    fig = draw_source_target_distributions_and_ratio(source_train_p, target_train_p,
         variables = drawing_variables,
         source_weights = source_train_p['init_wgt'],
         target_weights = target_train_p['weight'],
         new_source_weights = all_weights,
            legends = ['Source', 'Source (Reweighted)', 'Target'],
         # xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
         # ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
         # scale_target = scale_target_train,
         shape_only = True 
         )

    # add gloabal title to the figure
    fig.suptitle(f'Shape only. Process: {process} in category: {category}', fontsize=16)
    fig.savefig(f'{process_pics_folder}ReweightingResult_{process}_{category}_Shape.png')
    print(f"Saved reweighting result figure to ReweightingResult_{process}_{category}_Shape.png")
    plt.close()

    # Build per-event derived variables for psi-prime sliced diagnostics.
    # (already computed above; just reuse here)
    source_weights_np = source_train_p['init_wgt'].to_numpy()
    target_weights_np = target_train_p['weight'].to_numpy()

    print("Producing per-process psi-prime plots in muon pT slices...")
    pt_centers = 0.5 * (MUON_PT_BIN_EDGES_GEV[:-1] + MUON_PT_BIN_EDGES_GEV[1:])
    mean_source_vs_pt = []
    mean_target_vs_pt = []
    mean_reweighted_vs_pt = []
    for i in range(len(MUON_PT_BIN_EDGES_GEV) - 1):
        low = MUON_PT_BIN_EDGES_GEV[i]
        high = MUON_PT_BIN_EDGES_GEV[i + 1]
        if i == len(MUON_PT_BIN_EDGES_GEV) - 2:
            source_slice_mask = (
                (source_train_p['muon_pt_gev'].to_numpy() >= low)
                & (source_train_p['muon_pt_gev'].to_numpy() <= high)
            )
            target_slice_mask = (
                (target_train_p['muon_pt_gev'].to_numpy() >= low)
                & (target_train_p['muon_pt_gev'].to_numpy() <= high)
            )
        else:
            source_slice_mask = (
                (source_train_p['muon_pt_gev'].to_numpy() >= low)
                & (source_train_p['muon_pt_gev'].to_numpy() < high)
            )
            target_slice_mask = (
                (target_train_p['muon_pt_gev'].to_numpy() >= low)
                & (target_train_p['muon_pt_gev'].to_numpy() < high)
            )

        save_psi_prime_slice_plot(
            source_df=source_train_p,
            target_df=target_train_p,
            source_weights=source_weights_np,
            target_weights=target_weights_np,
            new_source_weights=all_weights,
            source_mask=source_slice_mask,
            target_mask=target_slice_mask,
            pics_folder_name=process_pics_folder,
            process=process,
            category=category,
            slice_type='pt',
            bin_index=i,
            low=low,
            high=high,
            unit='GeV',
        )

        mean_source_vs_pt.append(
            _hist_density_mean(
                source_train_p['psi_prime'].to_numpy()[source_slice_mask],
                source_weights_np[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_target_vs_pt.append(
            _hist_density_mean(
                target_train_p['psi_prime'].to_numpy()[target_slice_mask],
                target_weights_np[target_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_reweighted_vs_pt.append(
            _hist_density_mean(
                source_train_p['psi_prime'].to_numpy()[source_slice_mask],
                all_weights[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )

    save_mean_vs_slice_plot(
        x_centers=pt_centers,
        source_means=mean_source_vs_pt,
        target_means=mean_target_vs_pt,
        reweighted_means=mean_reweighted_vs_pt,
        x_label='Muon pT',
        slice_name='pt',
        unit='GeV',
        process=process,
        category=category,
        output_dir=process_pics_folder,
    )

    print("Producing per-process psi-prime plots in recoil slices...")
    recoil_centers = 0.5 * (RECOIL_BIN_EDGES_MEV[:-1] + RECOIL_BIN_EDGES_MEV[1:])
    mean_source_vs_recoil = []
    mean_target_vs_recoil = []
    mean_reweighted_vs_recoil = []
    for i in range(len(RECOIL_BIN_EDGES_MEV) - 1):
        low = RECOIL_BIN_EDGES_MEV[i]
        high = RECOIL_BIN_EDGES_MEV[i + 1]
        if i == len(RECOIL_BIN_EDGES_MEV) - 2:
            source_slice_mask = (
                (source_train_p['recoil_mev'].to_numpy() >= low)
                & (source_train_p['recoil_mev'].to_numpy() <= high)
            )
            target_slice_mask = (
                (target_train_p['recoil_mev'].to_numpy() >= low)
                & (target_train_p['recoil_mev'].to_numpy() <= high)
            )
        else:
            source_slice_mask = (
                (source_train_p['recoil_mev'].to_numpy() >= low)
                & (source_train_p['recoil_mev'].to_numpy() < high)
            )
            target_slice_mask = (
                (target_train_p['recoil_mev'].to_numpy() >= low)
                & (target_train_p['recoil_mev'].to_numpy() < high)
            )

        save_psi_prime_slice_plot(
            source_df=source_train_p,
            target_df=target_train_p,
            source_weights=source_weights_np,
            target_weights=target_weights_np,
            new_source_weights=all_weights,
            source_mask=source_slice_mask,
            target_mask=target_slice_mask,
            pics_folder_name=process_pics_folder,
            process=process,
            category=category,
            slice_type='recoil',
            bin_index=i,
            low=low,
            high=high,
            unit='MeV',
        )

        mean_source_vs_recoil.append(
            _hist_density_mean(
                source_train_p['psi_prime'].to_numpy()[source_slice_mask],
                source_weights_np[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_target_vs_recoil.append(
            _hist_density_mean(
                target_train_p['psi_prime'].to_numpy()[target_slice_mask],
                target_weights_np[target_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_reweighted_vs_recoil.append(
            _hist_density_mean(
                source_train_p['psi_prime'].to_numpy()[source_slice_mask],
                all_weights[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )

    save_mean_vs_slice_plot(
        x_centers=recoil_centers,
        source_means=mean_source_vs_recoil,
        target_means=mean_target_vs_recoil,
        reweighted_means=mean_reweighted_vs_recoil,
        x_label='Recoil',
        slice_name='recoil',
        unit='MeV',
        process=process,
        category=category,
        output_dir=process_pics_folder,
    )

    all_source_plot_chunks.append(pd.DataFrame({
        'psi_prime': source_train_p['psi_prime'].to_numpy(),
        'muon_pt_gev': source_train_p['muon_pt_gev'].to_numpy(),
        'recoil_mev': source_train_p['recoil_mev'].to_numpy(),
        'source_weight': source_weights_np,
        'reweighted_weight': all_weights,
    }))
    all_target_plot_chunks.append(pd.DataFrame({
        'psi_prime': target_train_p['psi_prime'].to_numpy(),
        'muon_pt_gev': target_train_p['muon_pt_gev'].to_numpy(),
        'recoil_mev': target_train_p['recoil_mev'].to_numpy(),
        'target_weight': target_weights_np,
    }))


    # Generate a TTree with branches: eventID, entryNumber, init_wgt, weight (weight after training)

    dict_process = {
        'eventID': source_train_p['eventID'],
        'originalTreeEntry': source_train_p['originalTreeEntry'],
        'init_wgt': source_train_p['init_wgt'],
        'weight': all_weights
    }

    # append to dict_to_tree
    for key in dict_process.keys():
        if key not in dict_to_tree:
            dict_to_tree[key] = []
        dict_to_tree[key].extend(dict_process[key])

    print(f"Total event rate before reweighting for process {process}: {source_n_events_before:.2f}")
    print(f"Total event rate after reweighting for process {process}: {source_n_events_after:.2f}")


all_process_pics_folder = f'{pics_folder_name}all_processes/'
os.makedirs(all_process_pics_folder, exist_ok=True)

if len(all_source_plot_chunks) > 0 and len(all_target_plot_chunks) > 0:
    all_source_plot = pd.concat(all_source_plot_chunks, ignore_index=True)
    all_target_plot = pd.concat(all_target_plot_chunks, ignore_index=True)

    all_source_weights = all_source_plot['source_weight'].to_numpy()
    all_reweighted_weights = all_source_plot['reweighted_weight'].to_numpy()
    all_target_weights = all_target_plot['target_weight'].to_numpy()

    print("Producing all-process psi-prime plots in muon pT slices...")
    pt_centers = 0.5 * (MUON_PT_BIN_EDGES_GEV[:-1] + MUON_PT_BIN_EDGES_GEV[1:])
    mean_source_vs_pt = []
    mean_target_vs_pt = []
    mean_reweighted_vs_pt = []
    for i in range(len(MUON_PT_BIN_EDGES_GEV) - 1):
        low = MUON_PT_BIN_EDGES_GEV[i]
        high = MUON_PT_BIN_EDGES_GEV[i + 1]
        if i == len(MUON_PT_BIN_EDGES_GEV) - 2:
            source_slice_mask = (
                (all_source_plot['muon_pt_gev'].to_numpy() >= low)
                & (all_source_plot['muon_pt_gev'].to_numpy() <= high)
            )
            target_slice_mask = (
                (all_target_plot['muon_pt_gev'].to_numpy() >= low)
                & (all_target_plot['muon_pt_gev'].to_numpy() <= high)
            )
        else:
            source_slice_mask = (
                (all_source_plot['muon_pt_gev'].to_numpy() >= low)
                & (all_source_plot['muon_pt_gev'].to_numpy() < high)
            )
            target_slice_mask = (
                (all_target_plot['muon_pt_gev'].to_numpy() >= low)
                & (all_target_plot['muon_pt_gev'].to_numpy() < high)
            )

        save_psi_prime_slice_plot(
            source_df=all_source_plot,
            target_df=all_target_plot,
            source_weights=all_source_weights,
            target_weights=all_target_weights,
            new_source_weights=all_reweighted_weights,
            source_mask=source_slice_mask,
            target_mask=target_slice_mask,
            pics_folder_name=all_process_pics_folder,
            process='all',
            category=category,
            slice_type='pt',
            bin_index=i,
            low=low,
            high=high,
            unit='GeV',
        )

        mean_source_vs_pt.append(
            _hist_density_mean(
                all_source_plot['psi_prime'].to_numpy()[source_slice_mask],
                all_source_weights[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_target_vs_pt.append(
            _hist_density_mean(
                all_target_plot['psi_prime'].to_numpy()[target_slice_mask],
                all_target_weights[target_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_reweighted_vs_pt.append(
            _hist_density_mean(
                all_source_plot['psi_prime'].to_numpy()[source_slice_mask],
                all_reweighted_weights[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )

    save_mean_vs_slice_plot(
        x_centers=pt_centers,
        source_means=mean_source_vs_pt,
        target_means=mean_target_vs_pt,
        reweighted_means=mean_reweighted_vs_pt,
        x_label='Muon pT',
        slice_name='pt',
        unit='GeV',
        process='all',
        category=category,
        output_dir=all_process_pics_folder,
    )

    print("Producing all-process psi-prime plots in recoil slices...")
    recoil_centers = 0.5 * (RECOIL_BIN_EDGES_MEV[:-1] + RECOIL_BIN_EDGES_MEV[1:])
    mean_source_vs_recoil = []
    mean_target_vs_recoil = []
    mean_reweighted_vs_recoil = []
    for i in range(len(RECOIL_BIN_EDGES_MEV) - 1):
        low = RECOIL_BIN_EDGES_MEV[i]
        high = RECOIL_BIN_EDGES_MEV[i + 1]
        if i == len(RECOIL_BIN_EDGES_MEV) - 2:
            source_slice_mask = (
                (all_source_plot['recoil_mev'].to_numpy() >= low)
                & (all_source_plot['recoil_mev'].to_numpy() <= high)
            )
            target_slice_mask = (
                (all_target_plot['recoil_mev'].to_numpy() >= low)
                & (all_target_plot['recoil_mev'].to_numpy() <= high)
            )
        else:
            source_slice_mask = (
                (all_source_plot['recoil_mev'].to_numpy() >= low)
                & (all_source_plot['recoil_mev'].to_numpy() < high)
            )
            target_slice_mask = (
                (all_target_plot['recoil_mev'].to_numpy() >= low)
                & (all_target_plot['recoil_mev'].to_numpy() < high)
            )

        save_psi_prime_slice_plot(
            source_df=all_source_plot,
            target_df=all_target_plot,
            source_weights=all_source_weights,
            target_weights=all_target_weights,
            new_source_weights=all_reweighted_weights,
            source_mask=source_slice_mask,
            target_mask=target_slice_mask,
            pics_folder_name=all_process_pics_folder,
            process='all',
            category=category,
            slice_type='recoil',
            bin_index=i,
            low=low,
            high=high,
            unit='MeV',
        )

        mean_source_vs_recoil.append(
            _hist_density_mean(
                all_source_plot['psi_prime'].to_numpy()[source_slice_mask],
                all_source_weights[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_target_vs_recoil.append(
            _hist_density_mean(
                all_target_plot['psi_prime'].to_numpy()[target_slice_mask],
                all_target_weights[target_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )
        mean_reweighted_vs_recoil.append(
            _hist_density_mean(
                all_source_plot['psi_prime'].to_numpy()[source_slice_mask],
                all_reweighted_weights[source_slice_mask],
                PSI_PRIME_BIN_EDGES,
            )
        )

    save_mean_vs_slice_plot(
        x_centers=recoil_centers,
        source_means=mean_source_vs_recoil,
        target_means=mean_target_vs_recoil,
        reweighted_means=mean_reweighted_vs_recoil,
        x_label='Recoil',
        slice_name='recoil',
        unit='MeV',
        process='all',
        category=category,
        output_dir=all_process_pics_folder,
    )


# sort dict_to_tree entries by originalTreeEntry
sorted_indices = np.argsort(dict_to_tree['originalTreeEntry'])
for key in dict_to_tree.keys():
    dict_to_tree[key] = np.array(dict_to_tree[key])[sorted_indices]

output_folder = pathlib.Path(target_path).parent
output_folder = output_folder / 'BDTreweight_outputs'
output_folder.mkdir(parents=True, exist_ok=True)
source_basename = pathlib.Path(source_path).stem
match = re.search(r'minervame..', source_basename)
playlist_name = match.group(0) if match else 'unknownPlaylist'

if (build_tree_of_weights):
    output_root_file = output_folder / f'ReweightWeights_{playlist_name}_{target_model_name}_{category}.root'
    with uproot.recreate(output_root_file) as f_out:
            f_out.mktree("reweight_tree",dict_to_tree)

    # check that the output file has been created and it's sorted out
    f_in = uproot.open(output_root_file)
    tree_in = f_in['reweight_tree']
    tree_in.show()
    # equivalent of Scan in ROOT
    print("")
    print(tree_in.arrays(library='pd'))

    print(f"Produced weights saved to {output_root_file}")
