#!/bin/bash

source="/Users/lorenzo/cernbox/MINERVA_MC/source//ReweightSourceCCQELike_ABCDEFGLMNOP.root"

# target="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
target="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"

# target_model_id="TEST-SF"
target_model_id="TEST-EDRMF"

module_path="/Users/lorenzo/Minerva/fork_reweighting/"

PYTHONPATH=${module_path}:${PYTHONPATH} 

python3 ${module_path}/BDTReweight/train_by_reaction.py \
                    --source_path $source \
                    --target_path $target \
                    --module_path $module_path \
                    --model_name ${target_model_id} # --max_events 10000