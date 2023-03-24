NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNDAT/DCNDATv1_shareDCNBwarpEmbT_QDCNAttnBothDAT_noPE_E5D10_distill_dim64_p256_bwarp \
--config configs/DCNDAT.yaml \
--world_size ${NUM_GPUS}