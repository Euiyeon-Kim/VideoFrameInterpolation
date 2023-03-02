NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name RSTTv1_ReconLossAll_G001_D001  \
--config configs/RSTT.yaml \
--world_size ${NUM_GPUS}