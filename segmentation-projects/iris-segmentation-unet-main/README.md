# Iris Segmentation U-net w/TPU (Dice Coef: 0.94, Jaccard Index : 0.88)

I have used the following methods.

* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation,
* The project took place using <b>Google TPU</b>,
* <b>Custom layers</b> for encoding and decoding,
* <b>Custom callback</b> class  that used predicting a sample from the train dataset during training

## Results for 40 epochs
* Test Dice Coefficient : <b>0.94</b>
* Test Jaccard Index : <b>0.88</b>

## Predictions 

![266796515-aa78767d-4f7f-4df3-a5eb-6cd8d943c7a2](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/ebe486db-ff6b-40f7-88cf-8725fd8666a1)


![266796487-4d1757b3-17b8-4c42-badc-ab3a19a57a06](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/c99a97a6-7efa-4045-a595-531973f767a3)




## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1505.04597
2. https://www.aao.org/eye-health/anatomy/parts-of-eye
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
