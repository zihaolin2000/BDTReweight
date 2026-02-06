print('Import libraries...')
import sys
# Change this path to your working directory where BDTReweight is installed:
sys.path.append('/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/')
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
# from BDTReweight.utilities import particle_variable_to_latex, diff_xsec_latex_wrt_variable
import numpy as np
import pandas as pd
print('Imported.')


# set train / test size: total events
# size = 1000000
size = 4000000


# prepare target sample NuisanceFlatTree for training
print('Load target MC train sample...')
tree_target_train = NuisanceFlatTree(
    # # GENEIE v3.04.00 AR23 sample:
    '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_NUISFLAT.root',
    entry_start=0, entry_stop=size
#     ['/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_500_NUISFLAT.root',
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_501_NUISFLAT.root',
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_502_NUISFLAT.root',
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_503_NUISFLAT.root'],
    # Alternatively, try GENIE v3 G18_10a:
    # '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v3_0_6/tracker/G18_10a_02_11a/CH/flat_GENIE_G18_10a_02_11a_50M.root',
    # Use akward array's kwargs to control sample size
)
mask_CCQELike = (tree_target_train.get_mask_final_state_allowed_pdg([13, 2212, 2112])
        & (tree_target_train._flattree_vars['tgta']==12)
        & tree_target_train.get_mask_topology({'proton':'>=1'}))
tree_target_train.update_tree_with_mask(mask_CCQELike)
# Turn off target MC's extended 2p2h to fix q3 phase space issue in 2p0n final states:
mask_safe_2p2h = ~((tree_target_train._flattree_vars['Mode'] == 2) & (tree_target_train._flattree_vars['q3'] > 1.2))
tree_target_train.update_tree_with_mask(mask_safe_2p2h)
tree_target_train._flattree_vars['dpt'] = tree_target_train._flattree_vars['dpt'] / 1000

print('Loaded.')

# Specify detecting thresholds and topology particle counts:
KE_thresholds={'proton':0.05, 'neutron':0.01}

# Create dictionaries to store dataframes later:
source_train = {}
target_train = {}

# Record a normalization factor
normalizations = {}

categories = ['0p0n', '0pNn', '1p0n', '1pNn', '2p0n', '2pNn', 'others']
# categories = ['1p0n']
save_normalization = False

reweight_variables_cat = {
    '0p0n':['total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz'],
    '0pNn':['leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz'],
    '1p0n':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz'],
    '1pNn':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz'],
    '2p0n':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            'subleading_proton_px', 'subleading_proton_py', 'subleading_proton_pz'],
    '2pNn':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_neutron_px', 'leading_neutron_py', 
            'leading_neutron_pz','leading_muon_py','leading_muon_pz', 
            'subleading_proton_px', 'subleading_proton_py', 'subleading_proton_pz'],
    'others':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz']
}

variable_to_extract = {
    '0p0n':['Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE'],
    '0pNn':['Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz', 'leading_neutron_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE'],
    '1p0n':['Enu_true', 'Q2', 'q0', 'q3', 'W',
            'dpt', 'dalphat', 'dphit',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
            'leading_proton_px', 'leading_proton_py', 'leading_proton_pz', 'leading_proton_KE'],
    '1pNn':['Enu_true', 'Q2', 'q0', 'q3', 'W',
            'dalphat', 'dpt', 'dphit',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
            'leading_proton_px', 'leading_proton_py', 'leading_proton_pz', 'leading_proton_KE',
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz', 'leading_neutron_KE'],
    '2p0n':['Enu_true', 'Q2', 'q0', 'q3', 'W', 'Mode',
            'dalphat', 'dpt', 'dphit',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
            'leading_proton_px', 'leading_proton_py', 'leading_proton_pz', 'leading_proton_KE',
            'subleading_proton_px', 'subleading_proton_py', 'subleading_proton_pz', 'subleading_proton_KE'],
    '2pNn':['Enu_true', 'Q2', 'q0', 'q3', 'W',
            'dalphat', 'dpt', 'dphit',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
            'leading_proton_px', 'leading_proton_py', 'leading_proton_pz', 'leading_proton_KE',
            'subleading_proton_px', 'subleading_proton_py', 'subleading_proton_pz', 'subleading_proton_KE',
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz', 'leading_neutron_KE'],
    'others':['Enu_true', 'Q2', 'q0', 'q3', 'W',
            'dalphat', 'dpt', 'dphit',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
            'leading_proton_px', 'leading_proton_py', 'leading_proton_pz', 'leading_proton_KE']
}

particles_transform_to_reaction_frame = {
    '0p0n':['total_proton'],
    '0pNn':['leading_neutron','total_proton'],
    '1p0n':['leading_proton','total_proton'],
    '1pNn':['leading_proton','leading_neutron','total_proton'],
    '2p0n':['leading_proton','subleading_proton','total_proton'],
    '2pNn':['leading_proton','subleading_proton','leading_neutron','total_proton'],
    'others':['leading_proton','total_proton']
}

particle_counts_cat = {
    '0p0n':{'muon':'==1', 'proton':'==0', 'neutron':'==0'},
    '0pNn':{'muon':'==1', 'proton':'==0', 'neutron':'>=1'},
    '1p0n':{'muon':'==1', 'proton':'==1', 'neutron':'==0'},
    '1pNn':{'muon':'==1', 'proton':'==1', 'neutron':'>=1'},
    '2p0n':{'muon':'==1', 'proton':'==2', 'neutron':'==0'},
    '2pNn':{'muon':'==1', 'proton':'==2', 'neutron':'>=1'},
    'others':{'muon':'==1', 'proton':'>=3'}
}

# ==================================== loop through categories and train ====================================
for category in categories:

    print(f'==================================== Train {category} ====================================')

    # Specify particle counts in final state:
    particle_counts = particle_counts_cat[category]

    # Specify reweight training variables:
    reweight_variables = reweight_variables_cat[category]

    # List variables to extract from tree:
    variable_exprs = variable_to_extract[category]

    # Specify selector_particle's whose momentum are transformed to reaction frame:
    # (leading muon px py pz is assumed to be transformed)
    particle_names = particles_transform_to_reaction_frame[category]

    # Create a mask for the topology and create dataframes:
    mask = tree_target_train.get_mask_topology(particle_counts = particle_counts, KE_thresholds = KE_thresholds)
    print(category, np.sum(mask))
