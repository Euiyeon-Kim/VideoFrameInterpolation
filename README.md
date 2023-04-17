# VideoFrameInterpolation
Unified personal research repository  
Some models in models.archive is not working in this framework


### To Try
- [ ] m2m Flow variance 가 낮아지게 학습해보기
- [ ] distillation 할 때 epe기준 말고 variance 기준 weighting (modify get robust weight)
- [ ] 좀 흐려보이는데 cosine positional encoding 추가해보기  
- [ ] RAFT 처럼 attention 할 수 있는 방법 찾아보기


## Models - Baseline

---
**IFRNet (4,959,692)**
Time: 0.006s
Parameters: 4.96M

**EMA-VFI small**
Time: 0.014s
Parameters: 14.49M

**EMA-VFI**
Time: 0.034s
Parameters: 65.66M


### DCNTrans 계열

---

- Deformable Conv로 Query building
- Source Target에 대해 다 attention하고 mixing
- Deformable attention 아니고 Swin attention임

**DCNTransv1 (2,715,457)**
- 원래 IFRNet은 Decoder4에서 바로 feature t 생성해버림
- 여기서는 DCN을 사용한 Query builder로 feat_t_3 생성
- GMTrans에 있는 decoder2 로 decoding

**DCNTransv1_decRes10_GeoF3_noDistill_halfTonly (5,509,399)**
- encoder residual layer 5개, decoder residual 10개
**DCNTransv1_sepDCN_E5D10_dim64_Geo32_distill_bwarp (4,255,319)**
**DCNTransv2_sepDCN_E5D10_dim64_Geo32_distill_featFwarp (4,255,319)**

### DAT 계열

---

**DATv1_sepDCNBwarpEmbT_shareAttBothDAT_noPE_E0D5_dim72_p256_bwarp (4,042,351)**
**DATv1_sepDCNBwarp_shareDAT_noPE_E5D10_dim72_bwarp (5,335,111)**
**DATv1_sepDCNBwarpEmbT_shareAttBothDAT_noPE_E5D10_dim72_bwarp(4,977,631)**



### DCNDAT 계열

---

**DCNDATv1_shareDCNBwarpEmbT_QDCNAttnBothDAT_noPE_E5D10_distill_dim64_p256_bwarp (3,751,637)**
Time: 0.048s  
Parameters: 3.75M

