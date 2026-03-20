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

# arguments parser
p = argparse.ArgumentParser(description='Train BDT reweighter by reaction channel.')
p.add_argument('--source_path', '-s', type=str, help='Path to the source model ROOT file.')
p.add_argument('--target_path', '-t', type=str, help='Path to the target model ROOT file.')
p.add_argument('--module_path', '-m', type=str, help='Path to the BDTReweight module.')
p.add_argument('--model_name', type=str, help='Identifier of the target model.')
p.add_argument('--build_tree_of_weights',action='store_true', help='Activate building a ROOT TTree with the reweighting weights.')
p.add_argument('--shape_only', action='store_true', help='Only reweight shape, do not change total cross section')

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
        exit

print(f'Reweighting to target model: {target_model_name}')


tree_source_train = uproot.open(source_path)['EventKinematics_truth'].arrays(library='pd')

topologies = {0:'0p0n',1:'0pNn',2:'1p0n',3:'1pNn',4:'2p0n',5:'2pNn',6:'others'}
tree_source_train['topology'] = tree_source_train['topology'].map(topologies)
tree_source_train = tree_source_train.rename(columns={'muon_py':'leading_muon_py', 'muon_pz':'leading_muon_pz',
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

pics_folder_name = args.module_path + "pics/" + target_model_name + "/"
os.makedirs(args.module_path + "pics/", exist_ok=True)
os.makedirs(pics_folder_name, exist_ok=True)

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


tree_target_train = NuisanceFlatTree(target_path)
target_train = {}
target_test = {}
# Specify detecting thresholds and topology particle counts:
KE_thresholds={'proton':50000, 'neutron':50000} # very large thresholds to effectively put everything in 0p0n samples
# scale_source_train = len(tree_target_train._flattree_vars)/len(tree_source_train)
scale_source_train = 1 # 2.489225788674492e-44
# The following factor is used to set the total xsec.
# It should be the ratio between the total xsec predicted by the target model over that predicted by the source model. (σ_target / σ_source)
scale_target_train = 1 # 1.84e-43

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

# if 'neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_' in target_path:
    # target_is_from_hadded = True
target_is_from_hadded = False
# this is a bit silly: since I did hadd on nuisance flat trees, the total xsec is multiplied by the number of files I hadded (10)
target_ccqelike_xsec = tree_target_train.get_total_xsec()
if target_is_from_hadded:
    target_ccqelike_xsec /= 10 # divide by the number of files I hadded to get the correct total xsec for the target model (weird NUISANCE behavior...)
print(f"Total CCQELike xsec from target model: {target_ccqelike_xsec*1e38:.2f} x 10^-38 cm^2")

scale_target_train = target_ccqelike_xsec / source_ccqelike_xsec

if args.shape_only:
    print('Ignoring total cross section and modifying only shape')
    scale_target_train = 1.0

# Category name:
category = '0p0n'
particle_counts = {'muon':'==1', 'proton':'==0', 'neutron':'==0'}
variable_exprs = [
    'Enu_true', 'Q2', 'q0', 'q3', 'W',
    'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
    'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
]
# reweight_variables=['total_proton_px','total_proton_py','total_proton_pz','total_proton_KE','leading_muon_py','leading_muon_pz']
reweight_variables=['total_proton_KE','leading_muon_py','leading_muon_pz']
# drawing_variables = ['total_proton_px','total_proton_py','total_proton_pz','total_proton_KE','leading_muon_py','leading_muon_pz', 'weight']
drawing_variables = ['total_proton_KE','leading_muon_py','leading_muon_pz', 'weight']
particle_names = ['total_proton']

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
target_total_event_rate = scale_target_train * np.sum(tree_target_train.get_weight())
target_qe_event_rate = scale_target_train * np.sum(tree_target_train.get_weight()[tree_target_train.get_mode()==1])
target_2p2h_event_rate = scale_target_train * np.sum(tree_target_train.get_weight()[tree_target_train.get_mode()==2])
target_resdis_event_rate = scale_target_train * np.sum(tree_target_train.get_weight()[tree_target_train.get_mode()>2])
print(f"SOURCE QE event rate:      {source_qe_event_rate:.0f} ({source_qe_event_rate/source_total_event_rate*100:.2f} % )")
print(f"SOURCE 2p2h event rate:    {source_2p2h_event_rate:.0f} ({source_2p2h_event_rate/source_total_event_rate*100:.2f} % )")
print(f"SOURCE RES+DIS event rate: {source_resdis_event_rate:.0f} ({source_resdis_event_rate/source_total_event_rate*100:.2f} % )")
print(f"TARGET QE event rate:      {target_qe_event_rate:.0f} ({target_qe_event_rate/target_total_event_rate*100:.2f} % )")
print(f"TARGET 2p2h event rate:    {target_2p2h_event_rate:.0f} ({target_2p2h_event_rate/target_total_event_rate*100:.2f} % )")
print(f"TARGET RES+DIS event rate: {target_resdis_event_rate:.0f} ({target_resdis_event_rate/target_total_event_rate*100:.2f} % )")


dict_to_tree = {}

for process in ['2p2h','QE','Oth']:
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

    source_train_p = source_train[category][source_mask]
    source_test_p = source_train_p.iloc[np.arange(0, int(len(source_train_p)/10),1)].copy()
    target_train_p = target_train[category]
    target_test_p = target_train_p.copy()

    print(f"Source sample shape: {source_train_p[reweight_variables].shape}")
    print(f"Target sample shape: {target_train_p[reweight_variables].shape}")

    print("Fitting reweighter...")
    reweighter = Reweighter(n_estimators=100, learning_rate=0.4, max_depth=4, min_samples_leaf=30, gb_args={'subsample': 1.0})
    reweighter.fit(original=source_train_p[reweight_variables], target=target_train_p[reweight_variables],
                   target_weight=target_train_p['weight'],
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
        legends = ['Source', 'Target', 'Source (Reweighted)'],
        # xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
        # ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
        # scale_target = scale_target_train
    )

    # add gloabal title to the figure
    fig.suptitle(f'Reweighting Result for process: {process} in category: {category}', fontsize=16)
    fig.savefig(f'{pics_folder_name}ReweightingResult_{process}_{category}.png')
    print(f"Saved reweighting result figure to ReweightingResult_{process}_{category}.png")
    plt.close()

    fig = draw_source_target_distributions_and_ratio(source_train_p, target_train_p,
         variables = drawing_variables,
         source_weights = source_train_p['init_wgt'],
         target_weights = target_train_p['weight'],
         new_source_weights = all_weights,
         legends = ['Source', 'Target', 'Source (Reweighted)'],
         # xlabels = [particle_variable_to_latex(var) for var in drawing_variables],
         # ylabels = [diff_xsec_latex_wrt_variable(var) for var in drawing_variables],
         # scale_target = scale_target_train,
         shape_only = True
         )

    # add gloabal title to the figure
    fig.suptitle(f'Shape only. Process: {process} in category: {category}', fontsize=16)
    fig.savefig(f'{pics_folder_name}ReweightingResult_{process}_{category}_Shape.png')
    print(f"Saved reweighting result figure to ReweightingResult_{process}_{category}_Shape.png")
    plt.close()


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
