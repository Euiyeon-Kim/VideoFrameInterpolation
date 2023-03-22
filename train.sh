NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DAT/DATv1_shareDCNBwarpEmbT_shareDAT_withPE_E5D10_noDistill_dim64_p256_bwarp \
--config configs/DAT.yaml \
--world_size ${NUM_GPUS}