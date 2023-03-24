NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DAT/DATv1_sepDCNBwarpEmbT_shareAttBothDAT_noPE_E0D5_dim72_bwarp \
--config configs/DAT.yaml \
--world_size ${NUM_GPUS}