# VideoFrameInterpolation

### To Try
- [ ] m2m Flow variance 가 낮아지게 학습해보기
- [ ] distillation 할 때 epe기준 말고 variance 기준 weighting (modify get robust weight)
- [ ] 좀 흐려보이는데 cosine positional encoding 추가해보기  
- [ ] RAFT 처럼 attention 할 수 있는 방법 찾아보기

### To Do
- [ ] Resume training
  - [ ] 모델 weight loading
  - [ ] logger start step 설정


## Models

---
**IFRNet (4,959,692)**
Time: 0.006s
Parameters: 4.96M

**DATv1 (4,959,692)**
Time: 0.039s
Parameters: 3.93M

**EMA-VFI small**
Time: 0.022s
Parameters: 14.49M

**EMA-VFI**
Time: 0.077s
Parameters: 65.66M

**IFRM2Mv1 (2,937,414)**
- Decoder4에서 f01_4, f10_4 예측
- f01_4, f10_4 이미지 downsample해서 bwarp -> z0_4, z1_4 예측
- Decoder 3, 2, 1에서 source_feat, source_bwarp_target, z_s 받아서 residual flow랑 residual z 예측
- Geometry loss 없음
- Decoder 1는 output residual flow 5개


**DCNTransv1 (2,715,457)**
- 원래 IFRNet은 Decoder4에서 바로 feature t 생성해버림
- 여기서는 DCN을 사용한 Query builder로 feat_t_3 생성
- GMTrans에 있는 decoder2 로 decoding

**DCNTransv1_decRes10_GeoF3_noDistill_halfTonly (5,509,399)**
- encoder residual layer 5개, decoder residual 10개

**DCNTransv1_sepDCN_E5D10_dim64_Geo32_distill_bwarp (4,255,319)**
**DCNTransv2_sepDCN_E5D10_dim64_Geo32_distill_featFwarp (4,255,319)**


### DCNTrans
- Deformable Conv로 Query building
- Source Target에 대해 다 attention하고 mixing
- Deformable attention 아니고 Swin attention임

### Notice
DCNTransv1_sepDCN_E5D10_dim64_Geo32_distill_bwarp 
DCNTransv2_sepDCN_E5D10_dim64_Geo32_distill_fwarp
이후로 postion encoding들어가고
encoder 크기 좀 작아지고
DCN Blending block 크기도 작아지고 마지막에 activation 빠짐

DCNTransv2_swinV2_sepDCNAvgFwarpD4_dim64_enc5dec5_GeoF32_Distill_halfTonly
DCNTransv2_sepDCN_dim64_E5D5_Geo32_distill_flowReversal

DCNTransv1_swinV2_sepDCNatD4_enc5dec10_GeoF32_Distill_halfTonly
DCNTransv1_sepDCN_E5D10_dim72_Geo32_distill_bwarp