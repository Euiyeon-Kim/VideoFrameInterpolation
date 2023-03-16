NUM_GPUS=1

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DAT/debug \
--config configs/DAT.yaml \
--world_size ${NUM_GPUS}