#!/bin/bash

# require one argument for the target model ID
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_model_id>"
    exit 1
fi

# check that the setup is correct
if [ -z "$MINERVAEXE" ]; then
  echo "Error: MINERVAEXE is not set. Please source the setup script."
  exit 1
fi

source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_ABCDEFGLMNOP.root"
# source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_minervame1A.root"

target_NEUTSF="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
target_NEUTEDRMF="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"

target_NuWro_LFG="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"
target_GENIEv3="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"


# set the target model ID from the command line argument
target_model_id="$1"

# select the target file based on the target model ID
case "$target_model_id" in 
  "NEUT-SF")
    target=${target_NEUTSF}
    ;;
  "NEUT-EDRMF")
    target=${target_NEUTEDRMF}
    ;;
  "NuWro-LFG")
    target=${target_NuWro_LFG}
    ;;
  "GENIEv3")
    target=${target_GENIEv3}
    ;;
  *)
    echo "Error: Unknown target model ID '$target_model_id'. Please use 'NEUT-SF', 'NEUT-EDRMF', 'NuWro-LFG', or 'GENIEv3'."
    exit 1
    ;;
esac

target_folder=$(dirname "$target")

python3 ${MINERVA}/BDTReweight/train_by_reaction.py \
                    --source_path $source \
                    --target_path $target \
                    --module_path ${MINERVA} \
                    --plots_dir ${target_folder}/plots_${target_model_id} \
                    --model_name ${target_model_id} 