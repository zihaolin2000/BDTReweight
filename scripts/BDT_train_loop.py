print('Import libraries...')
import sys
# Change this path to your working directory where BDTReweight is installed:
sys.path.append('/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/')
from BDTReweight.analysis import transform_momentum_to_reaction_frame, create_dataframe_from_nuisance
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
import numpy as np
import pandas as pd
import time
print('Imported.')

# set train / test size: total events
size = 1000000
# size = 4000000
train_size = 'train_size_1M'
# train_size = 'train_size_4M'

# prepare source sample NuisanceFlatTree for training
print('Load source MC train sample...')

tree_source_train = NuisanceFlatTree(
    # # GENIE v2.12.6 sample (prepared by Dan Ruterbories):
    # '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_100_ghep.root',
    # entry_start=0, entry_stop=size
    ['/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_900_ghep.root',
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_901_ghep.root',
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_902_ghep.root',
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_903_ghep.root'
    ],
)
# GENIE v2 CCQE-like final state has these particles: muon, proton, neutron, 25 MeV binding energy place holder
# Their pdg: 13, 2212, 2112, 2000000101
# GENIE v3 doesn't have this place holder.
mask_CCQELike = (tree_source_train.get_mask_final_state_allowed_pdg([13, 2212, 2112, 2000000101])
        & (tree_source_train._flattree_vars['tgta']==12)
        & tree_source_train.get_mask_topology({'proton':'>=1'}))
tree_source_train.update_tree_with_mask(mask_CCQELike)
tree_source_train._flattree_vars['dpt'] = tree_source_train._flattree_vars['dpt'] / 1000

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

# Drop FSI bug events for GENIE v2 train and test sets
indices_good_FSI = tree_source_train.get_indices_genie2_drop_fsibug_events()
tree_source_train.update_tree_with_mask(indices_good_FSI)

print('Loaded.')

# Specify detecting thresholds and topology particle counts:
KE_thresholds={'proton':0.05, 'neutron':0.01}

# Create dictionaries to store dataframes later:
source_train = {}
target_train = {}

# Record a normalization factor
normalizations = {}

categories = ['0p0n', '0pNn', '1p0n', '1pNn', '2p0n', '2pNn', 'others']
# categories = ['2p0n']
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

def reweighter_instance(category):
    if category == '1p0n':
        # return Reweighter(n_estimators=400,learning_rate=0.1, max_depth=4,min_samples_leaf=30, gb_args={'subsample': 1.0})
        return Reweighter(n_estimators=20,learning_rate=0.1, max_depth=30,min_samples_leaf=30, gb_args={'subsample': 1.0})
    else:
        return Reweighter(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_leaf=30, gb_args={'subsample': 1.0})

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
    mask = tree_source_train.get_mask_topology(particle_counts = particle_counts, KE_thresholds = KE_thresholds)
    source_train[category] = create_dataframe_from_nuisance(tree_source_train, variable_exprs=variable_exprs, mask=mask)

    mask = tree_target_train.get_mask_topology(particle_counts = particle_counts, KE_thresholds = KE_thresholds)
    target_train[category] = create_dataframe_from_nuisance(tree_target_train, variable_exprs=variable_exprs, mask=mask)

    target_train[category] = create_dataframe_from_nuisance(tree_target_train, variable_exprs=variable_exprs, mask=mask)
    


    # Convert to reaction frame:
    source_train[category] = transform_momentum_to_reaction_frame(source_train[category], selector_lepton='leading_muon', particle_names=particle_names)
    target_train[category] = transform_momentum_to_reaction_frame(target_train[category], selector_lepton='leading_muon', particle_names=particle_names)

    # Create a Reweighter (inherited from hep_ml.reweight.GBReweighter) instance: 
    reweighter = reweighter_instance(category)

    # Traing reweighter using source and target sample's reweight variables:
    start_time = time.perf_counter()
    reweighter.fit(source_train[category][reweight_variables], target_train[category][reweight_variables],
                   original_weight=None, target_weight=None,)
    end_time = time.perf_counter()
    print(f'Elapsed time: {(end_time-start_time):.3f} sec')

    # Save reweighter to a path:
#     reweighter.save_to_pickle(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/reweighter_MINERvA_ME_numuCarbon_CCQELike_GENIEv2_to_v3AR23_1mu{category}.pkl')
    reweighter.save_to_pickle(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/{train_size}/reweighter_MINERvA_ME_numuCarbon_CCQELike_GENIEv2_to_v3AR23_1mu{category}.pkl')

    total_w = reweighter.predict_weights(source_train[category][reweight_variables], original_weight=None)
    print('source train:', len(source_train[category]), 'target train:', len(target_train[category]),
        'total source_w:', np.sum(total_w))
    ratio = len(target_train[category])/np.sum(total_w)
    print('ratio:', ratio)
    normalizations[category] = ratio

    print(f'{category} trained.')

print('Training normalizations:', normalizations)

if save_normalization:                
    with open('/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/CCQELike_MINERvA_GENIEv2_to_v3AR23_topology_normalizations.txt', 'w') as f:
        for norm in normalizations.values():
            f.write(f"{norm}\n")

