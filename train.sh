NUM_GPUS=1

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNIFR/debug  \
--config configs/DCNIFR.yaml \
--world_size ${NUM_GPUS}