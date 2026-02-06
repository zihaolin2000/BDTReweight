print('Import libraries...')
import sys
# Change this path to your working directory where BDTReweight is installed:
sys.path.append('/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/')
from BDTReweight.analysis import transform_momentum_to_reaction_frame, create_dataframe_from_nuisance, draw_source_target_distributions_and_ratio
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
from BDTReweight.utilities import particle_variable_to_latex, diff_xsec_latex_wrt_variable
import numpy as np
import pandas as pd
print('Imported.')

# set train / test size: total events
size = 1000000
# size = 4000000

train_size = 'train_size_1M'
# train_size = 'train_size_4M'

# prepare source sample NuisanceFlatTree for training
print('Load source MC test sample...')

tree_source_test = NuisanceFlatTree(
#     # Statistically independent GENIE v2.12.6 sample (prepared by Dan Ruterbories):
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_101_ghep.root',
#     entry_start=0, entry_stop=size
    [
    '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_104_ghep.root',
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_105_ghep.root',
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_106_ghep.root',
#     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_107_ghep.root'
    ],
)
mask_CCQELike = (tree_source_test.get_mask_final_state_allowed_pdg([13, 2212, 2112, 2000000101])
        & (tree_source_test._flattree_vars['tgta']==12)
        & tree_source_test.get_mask_topology({'proton':'>=1'}))
tree_source_test.update_tree_with_mask(mask_CCQELike)
tree_source_test._flattree_vars['dpt'] = tree_source_test._flattree_vars['dpt'] / 1000

# prepare target sample NuisanceFlatTree for testing
print('Load target MC test sample...')

tree_target_test = NuisanceFlatTree(
#     # A GENEIE v3.04.00 AR23 sample:
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_NUISFLAT.root',
#     entry_start=0, entry_stop=size,
    ['/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_500_NUISFLAT.root',
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_501_NUISFLAT.root',
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_502_NUISFLAT.root',
#     '/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_503_NUISFLAT.root'
    ],
)
mask_CCQELike = (tree_target_test.get_mask_final_state_allowed_pdg([13, 2212, 2112])
        & (tree_target_test._flattree_vars['tgta']==12)
        & tree_target_test.get_mask_topology({'proton':'>=1'}))
tree_target_test.update_tree_with_mask(mask_CCQELike)
# Turn off target MC's extended 2p2h to fix q3 phase space issue in 2p0n final states:
mask_safe_2p2h = ~((tree_target_test._flattree_vars['Mode'] == 2) & (tree_target_test._flattree_vars['q3'] > 1.2))
tree_target_test.update_tree_with_mask(mask_safe_2p2h)
tree_target_test._flattree_vars['dpt'] = tree_target_test._flattree_vars['dpt'] / 1000

# Set conversion factor from event rate to cross-section:
scale_source_test = tree_source_test.get_conversion_factor_eventrate_to_xsec()
scale_target_test = tree_target_test.get_conversion_factor_eventrate_to_xsec() # * tree_length / size # use this when large tree is partially imported

# Drop FSI bug events for GENIE v2 test sets
indices_good_FSI = tree_source_test.get_indices_genie2_drop_fsibug_events()
tree_source_test.update_tree_with_mask(indices_good_FSI)

print('Loaded.')

# Specify detecting thresholds and topology particle counts:
KE_thresholds = {'proton':0.05, 'neutron':0.01}

# Create dictionaries to store dataframes later:
source_test = {}
target_test = {}

categories = ['0p0n', '0pNn', '1p0n', '1pNn', '2p0n', '2pNn', 'others']
# categories = ['2p0n']

# make_combined_plots = True
make_combined_plots = False

# weight normalizations
# with open("/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/CCQELike_MINERvA_GENIEv2_to_v3AR23_topology_normalizations.txt") as file:
# with open(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/{train_size}/CCQELike_MINERvA_GENIEv2_to_v3AR23_topology_normalizations.txt') as file:
#     normalization_factors = [float(line) for line in file if line.strip()]
normalization_factors = np.loadtxt(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/{train_size}/CCQELike_MINERvA_GENIEv2_to_v3AR23_topology_normalizations.txt')

normalizations = {}
normalizations['0p0n'] = normalization_factors[0]
normalizations['0pNn'] = normalization_factors[1]
normalizations['1p0n'] = normalization_factors[2]
normalizations['1pNn'] = normalization_factors[3]
normalizations['2p0n'] = normalization_factors[4]
normalizations['2pNn'] = normalization_factors[5]
normalizations['others'] = normalization_factors[6]


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
    '2p0n':['Enu_true', 'Q2', 'q0', 'q3', 'W',
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

drawing_variables_cat = {
    '0p0n':['total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            'Enu_true', 'Q2', 'q0', 'q3', 
            'weight'],
    '0pNn':['leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            'Enu_true', 'Q2', 'q0', 'q3', 
            'weight'],
    '1p0n':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            # 'total_proton_px','total_proton_py','total_proton_pz', 
            'dpt', 'dalphat', 'dphit', 
            'Enu_true', 'Q2', 'q0', 'q3', 
            'weight'],
    '1pNn':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'leading_neutron_px','leading_neutron_py','leading_neutron_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz',
            # 'total_proton_px','total_proton_py','total_proton_pz',
            'dpt', 'dalphat', 'dphit',
            'Enu_true', 'Q2', 'q0', 'q3', 
            'weight'],
    '2p0n':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'subleading_proton_px','subleading_proton_py','subleading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            # 'total_proton_px','total_proton_py','total_proton_pz', 
            'dpt', 'dalphat', 'dphit',
            'Enu_true', 'Q2', 'q0', 'q3',
            'weight'],
    '2pNn':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'subleading_proton_px','subleading_proton_py','subleading_proton_pz',
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz',
            # 'total_proton_px','total_proton_py','total_proton_pz',
            'dpt', 'dalphat', 'dphit',
            'Enu_true', 'Q2', 'q0', 'q3', 
            'weight'],
    'others':['leading_proton_px','leading_proton_py','leading_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 
            # 'total_proton_px','total_proton_py','total_proton_pz', 
            'dpt', 'dalphat', 'dphit',
            'Enu_true', 'Q2', 'q0', 'q3',
            'weight']
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

# Legends of source, source reweighted, and target to be labeled on plot:
legends = ['GENIE v2.12.6 ($v2$)', 'GENIE v2.12.6 reweighted ($v2\'$)', 'GENIE v3.0.6 AR23 ($v3$)']


# ==================================== loop through categories and train ====================================
for category in categories:
    print(f'==================================== Plot {category} ====================================')

    # Specify particle counts in final state:
    particle_counts = particle_counts_cat[category]

    # List variables to extract from tree:
    variable_exprs = variable_to_extract[category]

    # Specify reweight training variables:
    reweight_variables = reweight_variables_cat[category]

    # Specify selector_particle's whose momentum are transformed to reaction frame:
    # (leading muon px py pz is assumed to be transformed)
    particle_names = particles_transform_to_reaction_frame[category]

    # Create a mask for the topology and create dataframes:
    mask = tree_source_test.get_mask_topology(particle_counts = particle_counts, KE_thresholds = KE_thresholds)
    source_test[category] = create_dataframe_from_nuisance(tree_source_test, variable_exprs=variable_exprs, mask=mask)

    mask = tree_target_test.get_mask_topology(particle_counts = particle_counts, KE_thresholds = KE_thresholds)
    target_test[category] = create_dataframe_from_nuisance(tree_target_test, variable_exprs=variable_exprs, mask=mask)

    # Convert to reaction frame:
    source_test[category] = transform_momentum_to_reaction_frame(source_test[category], selector_lepton='leading_muon', particle_names=particle_names)
    target_test[category] = transform_momentum_to_reaction_frame(target_test[category], selector_lepton='leading_muon', particle_names=particle_names)

    # Load reweighter and normalizations from path:
#     reweighter = Reweighter.load_from_pickle(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/reweighter_MINERvA_ME_numuCarbon_CCQELike_GENIEv2_to_v3AR23_1mu{category}.pkl')
    reweighter = Reweighter.load_from_pickle(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/{train_size}/reweighter_MINERvA_ME_numuCarbon_CCQELike_GENIEv2_to_v3AR23_1mu{category}.pkl')

    # Set target weights to 1.0 for all events:
    target_test[category]['weight'] = 1.0

    # Specify variables to be plotted as histograms:
    drawing_variables = drawing_variables_cat[category]

    # Predict weights for source model:
    source_test[category]['weight'] = reweighter.predict_weights(
        source_test[category][reweight_variables], 
        original_weight=None
    ) * normalizations[category]
    print('source train:', len(source_test[category]), 'target train:', len(target_test[category]),
        'total source_w:', np.sum(source_test[category]['weight']) / normalizations[category], 
        'normalization: *', f'{normalizations[category]:.3f}')

    # Draw distribution histograms:
    mask = source_test[category]['weight'] < 100
    draw_source_target_distributions_and_ratio(source_test[category][mask], target_test[category], 
        variables = drawing_variables,legends=legends,
        source_weights = np.ones(len(source_test[category][mask])),
        target_weights = np.ones(len(target_test[category])), 
        new_source_weights = source_test[category][mask]['weight'],
        xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
        ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
        scale_source = scale_source_test, scale_target = scale_target_test,
        savepath = f'/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/ZihaoWD/plots/{category}-test.png',
        figshow=False
    )
    print(f'{category} plotted.')
    
# ======================== combine together ============================
if make_combined_plots:
        # ======================== combine >=1p together ============================
        print('plot all >=1p categories...')
        # Category'1p0n', '1pNn', '2p0n', '2pNn', 'others':
        drawing_variables = [
                'leading_proton_px','leading_proton_py','leading_proton_pz',
                'total_proton_KE','leading_muon_py','leading_muon_pz', 
                # 'total_proton_px','total_proton_py','total_proton_pz', 
                'dpt', 'dalphat', 'dphit', 
                'Enu_true', 'Q2', 'q0', 'q3', 
                'weight'
        ]

        source_test_leading_proton = pd.concat([
                source_test['1p0n'][drawing_variables], source_test['1pNn'][drawing_variables], source_test['2p0n'][drawing_variables],
                source_test['2pNn'][drawing_variables], source_test['others'][drawing_variables]
        ])
        
        target_test_leading_proton = pd.concat([
                target_test['1p0n'][drawing_variables], target_test['1pNn'][drawing_variables], target_test['2p0n'][drawing_variables],
                target_test['2pNn'][drawing_variables], target_test['others'][drawing_variables]
        ])

        # Draw distribution histograms:
        mask = source_test_leading_proton['weight'] < 100
        draw_source_target_distributions_and_ratio(source_test_leading_proton[mask], target_test_leading_proton, 
                variables = drawing_variables,legends=legends,
                source_weights = np.ones(len(source_test_leading_proton[mask])),
                target_weights = np.ones(len(target_test_leading_proton)), 
                new_source_weights = source_test_leading_proton[mask]['weight'],
                xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
                ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
                scale_source = scale_source_test, scale_target = scale_target_test,
                savepath = '/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/ZihaoWD/plots/combined-test.png',
                figshow=False
        )

        # ======================== combine all together ============================
        print('plot all categories...')

        # All categories.
        drawing_variables = [
                'total_proton_px','total_proton_py','total_proton_pz', 
                'total_proton_KE','leading_muon_py','leading_muon_pz',
                'Enu_true', 'Q2', 'q0', 'q3', 
                'weight'
        ]

        source_test_all = pd.concat([
                source_test['0p0n'][drawing_variables], source_test['0pNn'][drawing_variables],
                source_test['1p0n'][drawing_variables], source_test['1pNn'][drawing_variables], source_test['2p0n'][drawing_variables],
                source_test['2pNn'][drawing_variables], source_test['others'][drawing_variables]
        ])

        target_test_all = pd.concat([
                target_test['0p0n'][drawing_variables], target_test['0pNn'][drawing_variables],
                target_test['1p0n'][drawing_variables], target_test['1pNn'][drawing_variables], target_test['2p0n'][drawing_variables],
                target_test['2pNn'][drawing_variables], target_test['others'][drawing_variables]
        ])

        # Draw distribution histograms:
        mask = source_test_all['weight'] < 100
        draw_source_target_distributions_and_ratio(source_test_all[mask], target_test_all, 
                variables = drawing_variables,legends = legends,
                source_weights = np.ones(len(source_test_all[mask])),
                target_weights = np.ones(len(target_test_all)), 
                new_source_weights = source_test_all[mask]['weight'],
                xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
                ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
                scale_source = scale_source_test, scale_target = scale_target_test,
                savepath = '/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/ZihaoWD/plots/combined-sumPronton-test.png',
                figshow=False
        )

