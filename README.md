# VideoFrameInterpolation

## Models

---

**IFRM2Mv1**
- Decoder4에서 f01_4, f10_4 예측
- f01_4, f10_4 이미지 downsample해서 bwarp -> z0_4, z1_4 예측
- Decoder 3, 2, 1에서 source_feat, source_bwarp_target, z_s 받아서 residual flow랑 residual z 예측
- Geometry loss 없음
- Decoder 1는 output residual flow 5개



### To Do
- [ ] Resume training
  - [ ] 모델 weight loading
  - [ ] logger start step 설정
