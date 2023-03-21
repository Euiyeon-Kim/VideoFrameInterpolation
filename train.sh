NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DAT/DATv1_sepDCNBwarp_shareDAT_noPE_E5D10_dim72_bwarp \
--config configs/DAT.yaml \
--world_size ${NUM_GPUS}