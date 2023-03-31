NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name MADAT/MADATv1 \
--config configs/MADAT.yaml \
--world_size ${NUM_GPUS}