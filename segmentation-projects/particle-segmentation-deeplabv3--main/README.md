# Particle Segmentation Custom DeepLabv3+

I have used the following methods.

* I have implemented DeepLabv3+ stemmed from this Keras example[1,2],
* I converted the original masks to <b><span style="color:#e74c3c;"> binary masks</span></b> in the image preprocessing stage (I suppose that the original source of the dataset contains  tiff images or another type of 16 bit depth images.),
* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation[3],
* I used <b> <b><span style="color:#e74c3c;"> DeepLabv3+ implementation</span></b> with ResNet50 backbone,
* Used tf.data for input pipeline,
* <b>A Custom layer</b> for convolution operation,
* <b>A Custom layer</b> for Dilated Spatial Pyramid Pooling operation,


## Evaluation Results

| after 25 epochs  | Train (418 imgs) | Valid (63 imgs) | Test (47 imgs) |
|------------------|------------------|------------------|-----------------|
| Loss             | 0.0211           | 0.2282           | 0.1159          |
| Dice Loss        | 0.0200           | 0.0725           | 0.0613          |
| Dice Coefficient | 0.9800           | 0.9275           | <b>0.9387 </b>  |
| Jaccard Index    | 0.9608           | 0.8656           | <b>0.8869    </b>  |

 
## Test Set Predictions
![download (27)](https://github.com/john-fante/particle-segmentation-deeplabv3-/assets/50263592/737346c3-6233-4711-8050-c4936c9270b7)
![download (26)](https://github.com/john-fante/particle-segmentation-deeplabv3-/assets/50263592/0260e2b0-0c8c-463d-854c-9586ffe185e5)

    


## References
1. https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model
2. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1802.02611
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
