NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNTrans/DCNTransv1_GeoF3_OSL01wT50_halfTonly  \
--config configs/DCNTrans.yaml \
--world_size ${NUM_GPUS}