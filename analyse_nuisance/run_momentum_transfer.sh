
model_file_1="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
model_file_2="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"


python momentum_transfer.py \
  --inputs $model_file_1 $model_file_2 \
  --labels NEUT-SF NEUT-EDRMF \
  --outdir momentum_transfer_plots \
  --levels 0.05 0.1 0.3 0.6 0.9 
#   --pt-bins 0,0.025,0.050,0.075 \
#   --pt-bins 0,0.075,0.15,0.25,0.325,0.4,0.475,0.55,0.7,0.85,1.0,1.25  \