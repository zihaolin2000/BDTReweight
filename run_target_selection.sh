#!/bin/zsh

sample="CCQELike"
# input="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT.root"
# output="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_${sample}.root"

# filename_no_extension="neut_MINERvAflux_SF_nu_all_NUISFLAT"
# filename_no_extension="Flattened_GENIE_v2_12_6_DefaultMEC_numu_CH_0_ghep"
filename_no_extension="flat_GENIE_G18_10b_02_11a_50M"
# filename_no_extension="mnv_nu_flat_SF_neut_103"
# filename_no_extension="neut_MINERvAflux_EDRMF_nu_all_NUISFLAT"
# filename_no_extension="flat_NuWro_CH_LFG_v2109_50M"
# filename_no_extension="flat_NuWro_CH_SF_v2109_50M"


# input="/afs/cern.ch/work/l/lgiannes/private/T2K/NEUT/nuisance/output/${filename_no_extension}.root"
# output="/afs/cern.ch/work/l/lgiannes/private/T2K/NEUT/nuisance/output/${filename_no_extension}_${sample}.root"

input="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/${filename_no_extension}.root"
output="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/${filename_no_extension}_${sample}.root"

python3 target_selection.py --input_file $input --output_file $output --sample $sample -m 10000000 # DO NOT USE OPTION -m! It cancels the meaning of the fScaleFactor in NUISANCE ttrees
