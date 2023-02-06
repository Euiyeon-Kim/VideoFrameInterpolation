python -m torch.distributed.launch \
--nproc_per_node=1 \
train.py \
--world_size 1