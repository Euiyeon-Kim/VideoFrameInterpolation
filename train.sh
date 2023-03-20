NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNTrans/DCNTransv2_swinV2_sepDCNAvgFwarpD4_dim64_enc5dec5_GeoF32_Distill_halfTonly \
--config configs/DCNTrans.yaml \
--world_size ${NUM_GPUS}