NUM_GPUS=4

python -m torch.distributed.launch --use_env \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name new_DAT/DAT_dim72_p256_nS8-16-32_oS2-4-8_b12 \
--config configs/DAT.yaml
