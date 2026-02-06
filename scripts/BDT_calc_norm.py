print('Import libraries...')
import sys
# Change this path to your working directory where BDTReweight is installed:
sys.path.append('/exp/minerva/app/users/zihaolin/REWEIGHTworkdir/')
from BDTReweight.analysis import transform_momentum_to_reaction_frame, create_dataframe_from_nuisance
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
# from BDTReweight.utilities import particle_variable_to_latex, diff_xsec_latex_wrt_variable
import numpy as np
print('Imported.')

train_size = 'train_size_1M'
# train_size = 'train_size_4M'

target_train_magnitude = {
    '0p0n':9703/4,
#     '0pNn':26721/4,
#     '1p0n':141665/4,
#     '1pNn':72661/4,
#     '2p0n':45068/4,
#     '2pNn':47838/4,
#     'others':60532/4,
# After q3 2p2h 1.2GeV cut: 
    '0pNn':26345/4,
    '1p0n':141274/4,
    '1pNn':67066/4,
    '2p0n':37774/4,
    '2pNn':43309/4,
    'others':56330/4,
}
# 4M:
# 0p0n: 0.5855641548628767
# 0pNn: 1.0014356690554638
# 1p0n: 1.1379473964553846
# 1pNn: 1.1886703696327063
# 2p0n: 1.6303624099301048
# 2pNn: 1.0279948237331964
# others: 0.8920484733711436

# 1M ({'0p0n': 0.663980738734788, '0pNn': 1.0448909510671904, '1p0n': 1.9591915707902254, 
# '1pNn': 1.21582406007389, '2p0n': 1.6893132112996045, '2pNn': 1.0831229924786883, 'others': 0.927843237453098})
# 0.5415912836729979
# 0.9546506566532239
# 1.1277639512716917
# 1.1835810806768043
# 1.6012229387281034
# 1.004431580791097
# 0.8871535984525127


means = []
categories = ['0p0n', '0pNn', '1p0n', '1pNn', '2p0n', '2pNn', 'others']
# categories = ['2p0n']
for category in categories:
        ratios = []
        # Load reweighter and normalizations from path:
        # reweighter = Reweighter.load_from_pickle(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/reweighter_MINERvA_ME_numuCarbon_CCQELike_GENIEv2_to_v3AR23_1mu{category}.pkl')
        reweighter = Reweighter.load_from_pickle(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/{train_size}/reweighter_MINERvA_ME_numuCarbon_CCQELike_GENIEv2_to_v3AR23_1mu{category}.pkl')

        for file_ghep in range(830,838):
                # for file_ghep in range(104,109):

                # set train / test size: total events
                size = 1000000
                # size = 4000000

                # prepare source sample NuisanceFlatTree for training
                print('Load source MC test sample...')

                tree_source_test = NuisanceFlatTree(
                        # Statistically independent GENIE v2.12.6 sample (prepared by Dan Ruterbories):
                        f'/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_{file_ghep}_ghep.root',
                        entry_start=0, entry_stop=size
                #     [
                #     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_104_ghep.root',
                #     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_105_ghep.root',
                #     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_106_ghep.root',
                #     '/pnfs/minerva/persistent/Models/GENIE/Medium_Energy/FHC/v2_12_6/tracker/minervabase/Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_107_ghep.root'
                #     ],
                )
                mask_CCQELike = (tree_source_test.get_mask_final_state_allowed_pdg([13, 2212, 2112, 2000000101])
                        & (tree_source_test._flattree_vars['tgta']==12)
                        & tree_source_test.get_mask_topology({'proton':'>=1'}))
                tree_source_test.update_tree_with_mask(mask_CCQELike)
                tree_source_test._flattree_vars['dpt'] = tree_source_test._flattree_vars['dpt'] / 1000


                # Drop FSI bug events for GENIE v2 test sets
                indices_good_FSI = tree_source_test.get_indices_genie2_drop_fsibug_events()
                tree_source_test.update_tree_with_mask(indices_good_FSI)

                print(f'File ghep {file_ghep} loaded.')

                # Specify detecting thresholds and topology particle counts:
                KE_thresholds = {'proton':0.05, 'neutron':0.01}

                # Create dictionaries to store dataframes later:
                source_test = {}

                make_combined_plots = False

                # weight normalizations
                with open("/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/CCQELike_MINERvA_GENIEv2_to_v3AR23_topology_normalizations.txt") as file:
                        normalization_factors = [float(line) for line in file if line.strip()]


                normalizations = {}
                normalizations['0p0n'] = normalization_factors[0]
                normalizations['0pNn'] = normalization_factors[1]
                # 1p0n: 4M GENIE v3 -> 141665/4 1p0n events. Corresponding 1M GENIE v2 was predicted with weights:
                # 130, 131, 132, 133, 134, 135, 136, 137
                # 31258.16514007156, 30842.794319281158, 31090.2317143042, 31104.39769889975, 31096.843495453046, 31203.2849985774, 31536.409100362755, 30862.28926245012 
                # mean: 31124.301966174997
                # "Empirical 1p0n ratio": 1.1378970053204525
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
                                # 'Enu_true', 'Q2', 'q0', 'q3', 
                                'weight'],
                        '0pNn':['leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
                                'total_proton_px','total_proton_py','total_proton_pz',
                                'total_proton_KE','leading_muon_py','leading_muon_pz', 
                                # 'Enu_true', 'Q2', 'q0', 'q3', 
                                'weight'],
                        '1p0n':['leading_proton_px','leading_proton_py','leading_proton_pz',
                                'total_proton_KE','leading_muon_py','leading_muon_pz', 
                                # 'total_proton_px','total_proton_py','total_proton_pz', 
                                'dpt', 'dalphat', 'dphit', 
                                # 'Enu_true', 'Q2', 'q0', 'q3', 
                                'weight'],
                        '1pNn':['leading_proton_px','leading_proton_py','leading_proton_pz',
                                'leading_neutron_px','leading_neutron_py','leading_neutron_pz',
                                'total_proton_KE','leading_muon_py','leading_muon_pz',
                                # 'total_proton_px','total_proton_py','total_proton_pz',
                                'dpt', 'dalphat', 'dphit',
                                # 'Enu_true', 'Q2', 'q0', 'q3', 
                                'weight'],
                        '2p0n':['leading_proton_px','leading_proton_py','leading_proton_pz',
                                'subleading_proton_px','subleading_proton_py','subleading_proton_pz',
                                'total_proton_KE','leading_muon_py','leading_muon_pz', 
                                # 'total_proton_px','total_proton_py','total_proton_pz', 
                                'dpt', 'dalphat', 'dphit',
                                # 'Enu_true', 'Q2', 'q0', 'q3',
                                'weight'],
                        '2pNn':['leading_proton_px','leading_proton_py','leading_proton_pz',
                                'subleading_proton_px','subleading_proton_py','subleading_proton_pz',
                                'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
                                'total_proton_KE','leading_muon_py','leading_muon_pz',
                                # 'total_proton_px','total_proton_py','total_proton_pz',
                                'dpt', 'dalphat', 'dphit',
                                # 'Enu_true', 'Q2', 'q0', 'q3', 
                                'weight'],
                        'others':['leading_proton_px','leading_proton_py','leading_proton_pz',
                                'total_proton_KE','leading_muon_py','leading_muon_pz', 
                                # 'total_proton_px','total_proton_py','total_proton_pz', 
                                'dpt', 'dalphat', 'dphit',
                                # 'Enu_true', 'Q2', 'q0', 'q3',
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

                print(f'==================================== calc {category} ====================================')

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


                # Convert to reaction frame:
                source_test[category] = transform_momentum_to_reaction_frame(source_test[category], selector_lepton='leading_muon', particle_names=particle_names)


                # Predict weights for source model:
                source_test[category]['weight'] = reweighter.predict_weights(
                        source_test[category][reweight_variables], 
                        original_weight=None
                )
                mask = source_test[category]['weight'] < 100
                print('source train:', len(source_test[category]), 'target train:', target_train_magnitude[category],
                        'total source_w:', np.sum(source_test[category]['weight']), 
                        'total <100 source_w:', np.sum(source_test[category][mask]['weight']), 
                        'normalization: *', f'{normalizations[category]:.3f}')
                ratios.append(target_train_magnitude[category] / np.sum(source_test[category][mask]['weight']))
        print('found ratios:',ratios, '\nmean:', np.mean(ratios))
        means.append(np.mean(ratios))

print('mean ratio:', means)
means = np.array(means)
np.savetxt(f'/exp/minerva/data/users/zihaolin/BDTReweighters/saved_reweighters_pickle/{train_size}/CCQELike_MINERvA_GENIEv2_to_v3AR23_topology_normalizations.txt',
        means, fmt='%.10f')
