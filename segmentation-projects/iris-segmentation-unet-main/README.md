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

![__results___28_1](https://github.com/john-fante/iris-segmentation-u-net/assets/50263592/4d1757b3-17b8-4c42-badc-ab3a19a57a06)
![__results___28_0](https://github.com/john-fante/iris-segmentation-u-net/assets/50263592/aa78767d-4f7f-4df3-a5eb-6cd8d943c7a2)


## Graphs

![__results___23_1](https://github.com/john-fante/iris-segmentation-u-net/assets/50263592/851bf393-6474-4bb0-b8a0-4b20e4344c7c)




## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1505.04597
2. https://www.aao.org/eye-health/anatomy/parts-of-eye
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
