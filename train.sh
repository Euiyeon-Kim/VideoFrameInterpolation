NUM_GPUS=1

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name debug \
--config configs/GMTrans.yaml \
--world_size ${NUM_GPUS}