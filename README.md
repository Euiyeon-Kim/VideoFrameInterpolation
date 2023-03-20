# VideoFrameInterpolation

### To Try
- [ ] m2m Flow variance 가 낮아지게 학습해보기
- [ ] distillation 할 때 epe기준 말고 variance 기준 weighting (modify get robust weight)

## Models

---
**IFRNet (4,959,692)**


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

### To Do
- [ ] Resume training
  - [ ] 모델 weight loading
  - [ ] logger start step 설정
