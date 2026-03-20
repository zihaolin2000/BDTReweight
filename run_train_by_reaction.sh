#!/bin/bash

# check that the setup is correct
if [ -z "$MINERVAEXE" ]; then
  echo "Error: MINERVAEXE is not set. Please source the setup script."
  exit 1
fi

source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_ABCDEFGLMNOP.root"
# source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_minervame1A.root"

target="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
# target="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"
# target="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/mnv_nu_flat_SF_neut_103_CCQELike.root"
# target="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"
# target="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"
target_model_id="NEUT-SF"

python3 ${MINERVA}/BDTReweight/train_by_reaction.py \
                    --source_path $source \
                    --target_path $target \
                    --module_path ${MINERVA} \
                    --model_name ${target_model_id} 
