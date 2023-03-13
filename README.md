# VideoFrameInterpolation

### To Try
- [ ] m2m Flow variance 가 낮아지게 학습해보기
- [ ] distillation 할 때 epe기준 말고 variance 기준 weighting (modify get robust weight)

## Models

---
**IFRM2Mv1 (2,937,414)**
- Decoder4에서 f01_4, f10_4 예측
- f01_4, f10_4 이미지 downsample해서 bwarp -> z0_4, z1_4 예측
- Decoder 3, 2, 1에서 source_feat, source_bwarp_target, z_s 받아서 residual flow랑 residual z 예측
- Geometry loss 없음
- Decoder 1는 output residual flow 5개


**DCNIFRv1 (?)**
- 원래 IFRNet은 Decoder4에서 바로 feature t 생성해버림
- 여기서는 Query builder로 feat_t_4 생성


### To Do
- [ ] Resume training
  - [ ] 모델 weight loading
  - [ ] logger start step 설정
