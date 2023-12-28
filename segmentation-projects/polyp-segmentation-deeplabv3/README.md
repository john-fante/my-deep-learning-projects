## Polyp Segmentation w/Custom DeepLabv3+(Dice: 0.77)

(kaggle link -> https://www.kaggle.com/code/banddaniel/polyp-segmentation-w-custom-deeplabv3-dice-0-77)

I have used the following methods.

* I have implemented DeepLabv3+ stemmed from this Keras example[1,2],
* I converted the original masks to <b><span style="color:#e74c3c;"> binary masks and applied dilated operation </span></b> in the image preprocessing stage,
* I used <b><span style="color:#e74c3c;"> histogram equalization and smoothing</span></b> for original images in the image preprocessing stage,
* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation[3],
* I used <b> <b><span style="color:#e74c3c;"> DeepLabv3+ implementation</span></b> with ResNet50 backbone,
* Used tf.data for input pipeline,
* <b>A Custom layer</b> for convolution operation,
* <b>A Custom layer</b> for Dilated Spatial Pyramid Pooling operation,
* <b>Custom callback</b> for predicting one sample from test dataset during training each epochs[4]

## Evaluation Results

| for 15 epochs    | Train (900 imgs) | Valid (135 imgs) | Test (100 imgs) |
|------------------|------------------|------------------|-----------------|
| Loss             | 0.0357           | 0.0399           | 0.2296          |
| Dice Loss        | 0.0657           | 0.0693           | 0.2111          |
| Dice Coefficient | 0.9343           | 0.9307           | 0.7889          |
| Jaccard Index    | 0.8770           | 0.8709           | 0.6601          |

    
## Model Improvement During Training
![download (1)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/59118b55-8814-4166-9254-43eb46017eba)


    
## Predictions
![download (43)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/994cadf0-73ce-41be-abc4-f40a02829a65)
![download (42)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/be597968-e29d-48ed-b055-a1a88ee9b476)



## References
1. https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model
2. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1802.02611
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
4. My another custom callbacks for Tensorflow (https://github.com/john-fante/my-tensorflow-custom-callbacks)
