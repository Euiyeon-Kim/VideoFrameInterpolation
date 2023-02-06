NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name IFRNet_baseline \
--config configs/IFRNet.yaml \
--world_size ${NUM_GPUS}