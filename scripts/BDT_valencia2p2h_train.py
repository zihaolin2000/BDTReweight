# BDT reweight training and saving to json.
# flux and target: T2K ND280 numu flux carbon; flux reweighted to BNB
# source MC: GENIE v3 AR23 SuSAv2 inclusive 2p2h events.
# target MC: Valencia exclusive 2p2h events.
# This file focus on pp final states.
# json reweighters will be read by nusystematics for future LArTPC experiment reweight.
# - Zihao Lin 2026 Aug 3


import sys
sys.path.append('/exp/icarus/app/users/zihaolin/REWEIGHTworkdir/')
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.analysis import transform_momentum_to_reaction_frame, draw_source_target_distributions_and_ratio, create_dataframe_from_nuisance
from BDTReweight.reweighter import Reweighter
from BDTReweight.utilities import particle_mass_lookup
from BDTReweight.utilities import particle_variable_to_latex, diff_xsec_latex_wrt_variable
import uproot
import awkward as ak
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from BDTReweight.json_reweighter import JSONReweighter, export_reweighter_json


target_vars = [
    'muon_Px', 'muon_Py', 'muon_Pz',
    'nucleon1_out_Px', 'nucleon1_out_Py','nucleon1_out_Pz',
    'nucleon2_out_Px','nucleon2_out_Py','nucleon2_out_Pz', 
    'weight_pp']


vars = [ 
    'leading_muon_px', 'leading_muon_py', 'leading_muon_pz',
    'leading_proton_px', 'leading_proton_py', 'leading_proton_pz',
    'subleading_proton_px', 'subleading_proton_py', 'subleading_proton_pz',
]


size = 1000000
genie_seed = 501
exclusive_size_pp = 25000

# prepare Valencia exclusive 2p2h dataframe
df_target = uproot.open('/exp/icarus/data/users/zihaolin/MC_outputs/Valencia2p2hsamples/MINERvAflux/KinematicVariables_Enu_Minerva_Exclusive.root'
    )['T'].arrays(library='pd',entry_start=0, entry_stop=exclusive_size_pp)[target_vars]
w_target = df_target['weight_pp'].copy()
df_target = df_target[['muon_Px', 'muon_Py', 'muon_Pz',
    'nucleon1_out_Px', 'nucleon1_out_Py','nucleon1_out_Pz',
    'nucleon2_out_Px','nucleon2_out_Py','nucleon2_out_Pz']].copy()
df_target.columns = vars
df_target['weight'] = w_target
df_target = df_target.loc[df_target['weight']>=0]
# FIXME: address T2K to BNB flux reweight.


for var in vars:
    df_target[var] = df_target[var]/1000 # MeV to GeV
df_target = transform_momentum_to_reaction_frame(df_target, 
    selector_lepton='leading_muon',
    particle_names=['leading_proton','subleading_proton'])
df_target['weight'] = df_target['weight'] * len(df_target) / np.sum(df_target['weight'])





# read GENIE v3 tree
tree_genie = NuisanceFlatTree(
    #f'/exp/minerva/data/users/zihaolin/MC_outputs/GENIE/GENIEv3_AR23_MINERvA_ME_FHC_numu_CH_{geniev3_seed}_NUISFLAT.root',
    f'/exp/icarus/data/users/zihaolin/MC_outputs/GENIE_v3/GENIEv3_AR23_MINERvA_ME_FHC_numu_C12_{genie_seed}_NUISFLAT.root',
    entry_start = 0, entry_stop = size)

# drop Enu > 20 GeV events for now
mask_2p2h = (tree_genie._flattree_vars['Enu_true'] <= 20.0) & (tree_genie._flattree_vars['Mode'] == 2) & (tree_genie._flattree_vars['nvertp'] == 6)
# entries_2p2h = np.arange(0, len(mask_2p2h), 1)[mask_2p2h]
tree_genie.update_tree_with_mask(mask_2p2h)

vertp_pdg = tree_genie._flattree_vars['pdg_vert'][:,3:5]
vertp_E = tree_genie._flattree_vars['E_vert'][:,3:5]
vertp_px = tree_genie._flattree_vars['px_vert'][:,3:5]
vertp_py = tree_genie._flattree_vars['py_vert'][:,3:5]
vertp_pz = tree_genie._flattree_vars['pz_vert'][:,3:5]


vertp_mass = ak.where(vertp_pdg == 2212, particle_mass_lookup('proton'),
    ak.where(vertp_pdg == 2112, particle_mass_lookup('neutron'), -999.0)
)
vertp_KE = vertp_E - vertp_mass
order = ak.argsort(vertp_KE, axis=1, ascending=False)
primary_pdg = vertp_pdg[order][:,0]
spectator_pdg = vertp_pdg[order][:,1]

mask = (primary_pdg == 2212) & (spectator_pdg == 2212) # pp final states
# mask = ~ ((primary_pdg == 2212) & (spectator_pdg == 2212)) # pn final states



# prepare SuSAv2 2p2h dataframe
df_source = pd.DataFrame() 
df_source['leading_muon_px'] = tree_genie._flattree_vars['px_vert'][:,5][mask]
df_source['leading_muon_py'] = tree_genie._flattree_vars['py_vert'][:,5][mask]
df_source['leading_muon_pz'] = tree_genie._flattree_vars['pz_vert'][:,5][mask]
 
# pp channel proton kinematics:
leading_i = ak.argmax(vertp_KE, axis=1, keepdims=True)
subleading_i = ak.argmin(vertp_KE, axis=1, keepdims=True)

df_source['leading_proton_px'] = vertp_px[leading_i][mask]
df_source['leading_proton_py'] = vertp_py[leading_i][mask]
df_source['leading_proton_pz'] = vertp_pz[leading_i][mask]
df_source['subleading_proton_px'] = vertp_px[subleading_i][mask]
df_source['subleading_proton_py'] = vertp_py[subleading_i][mask]
df_source['subleading_proton_pz'] = vertp_pz[subleading_i][mask]

df_source = transform_momentum_to_reaction_frame(df_source, 
    selector_lepton='leading_muon',
    particle_names=['leading_proton','subleading_proton'])



# reweight train and save
reweight_vars = [
    'leading_proton_px', 'leading_proton_py', 'leading_proton_pz',
    'subleading_proton_px', 'subleading_proton_py', 'subleading_proton_pz',
    'leading_muon_py', 'leading_muon_pz'
]


reweighter = Reweighter(n_estimators=200, learning_rate=0.1, max_depth=4, min_samples_leaf=20, gb_args={'subsample': 1.0})
print(f'size df_source: {len(df_source)}, target: {len(df_target)}')
reweighter.fit(df_source[reweight_vars], 
    df_target[reweight_vars],
    target_weight=df_target['weight']
)

export_reweighter_json(reweighter, f'/exp/icarus/data/users/zihaolin/MC_outputs/bdtreweighters_json/BDTReweighter_MINERvA_ME_numu_FHC_{genie_seed}_SuSAv2_to_ValenciaExclusive.json')





