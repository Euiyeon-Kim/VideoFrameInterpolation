NUM_GPUS=2

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name GMTransv1_startFromNoise_lossOnBaseFrame \
--config configs/GMTrans.yaml \
--world_size ${NUM_GPUS}