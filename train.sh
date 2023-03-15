NUM_GPUS=4

python -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name DCNTrans/DCNTransv1_swinV2_sepDCNatD4_enc5dec10_GeoF32_Distill_halfTonly \
--config configs/DCNTrans.yaml \
--world_size ${NUM_GPUS}