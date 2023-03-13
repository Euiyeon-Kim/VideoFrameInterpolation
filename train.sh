NUM_GPUS=1

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNTrans/DCNTransv1_GeoF3_noDistill_halfTonly  \
--config configs/DCNTrans.yaml \
--world_size ${NUM_GPUS}