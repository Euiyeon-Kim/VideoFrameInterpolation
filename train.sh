NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DAT/DATv1_sepDCNBwarp_shareDAT_noPE_E5D10_dim64_bwarp \
--config configs/DAT.yaml \
--world_size ${NUM_GPUS}