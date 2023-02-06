NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name IFRM2Mv1_nB5_noGeo \
--config configs/IFRM2M.yaml \
--world_size ${NUM_GPUS}