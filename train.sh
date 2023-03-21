NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNTrans/DCNTransv2_sepDCN_E5D10_dim64_Geo32_distill_fwarp \
--config configs/DCNTrans.yaml \
--world_size ${NUM_GPUS}
