## Lung Segmentation UNet w/SeparableConv

(kaggle link -> https://www.kaggle.com/code/banddaniel/lung-segmentation-unet-w-separableconv-dice-0-93)

I have used the following methods.

* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation[2],
* The project took place using <b>Google TPU</b>,
* There are many missing images (96 images). I used only mask images and correspondent images,
* I used <b>SeparableConv2D</b> instead of Conv2D in U-Net architecture,
* Used tf.data for input pipeline,
* <b>Custom layers</b> for encoding and decoding,
* <b>Custom callback</b> for predicting one sample from test dataset during training each 20 epochs[3]

## Evaluation Results (for 100 epochs)


|                  | Train (562 images) | Test (141 images) |
|------------------|--------------------|-------------------|
| Loss             | 0.0612             | 0.0697            |
| Dice Loss        | 0.0627             | 0.0684            |
| Dice Coefficient | 0.9373             | 0.9316            |
| Jaccard Index    | 0.8619             | 0.8588            |


## Model Improvement During Training
![download](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/7bff387e-6e4e-4e2a-ae74-f38288e2cec5)


## Predictions
![download (37)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/f47e4a27-9ccb-4b58-8962-fc4bbf869e94)



## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1505.04597
2. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
3. My another custom callbacks for Tensorflow (https://github.com/john-fante/my-tensorflow-custom-callbacks)
