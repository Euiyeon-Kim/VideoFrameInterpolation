NUM_GPUS=1

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNTrans/DCNTransv1_sepDCN_E5D5_dim64_Geo32_distill_bwarp \
--config configs/DCNTrans.yaml \
--world_size ${NUM_GPUS}