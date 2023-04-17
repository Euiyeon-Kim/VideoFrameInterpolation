NUM_GPUS=1

python -m torch.distributed.launch --use_env \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNDAT/debug \
--config configs/DCNDAT.yaml
