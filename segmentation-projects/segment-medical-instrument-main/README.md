# Segment Medical Instrument w/Custom DeepLabv3+

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/segment-medical-instrument-deeplabv3-dice-0-86 </b>

I have used the following methods.

* I have implemented DeepLabv3+ stemmed from this Keras example[1,2],
* I converted the original masks to <b><span style="color:#e74c3c;"> binary masks and applied dilated operation </span></b> in the image preprocessing stage,
* I used <b><span style="color:#e74c3c;"> histogram equalization and smoothing</span></b> for original images in the image preprocessing stage,
* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation[3],
* I used <b><span style="color:#e74c3c;"> DeepLabv3+ implementation</span></b> with ResNet50 backbone,
* Used tf.data for input pipeline,
* <b>A Custom layer</b> for convolution operation,
* <b>A Custom layer</b> for Dilated Spatial Pyramid Pooling operation,

    
## Evaluation Results

| after 25 epochs  | Train (531 imgs) | Valid (80 imgs) | Test (59 imgs) |
|------------------|------------------|------------------|-----------------|
| Loss             | 0.0071           | 0.0079           | 0.0861          |
| Dice Loss        | 0.0219           | 0.0239           | 0.0987          |
| Dice Coefficient | 0.9781           | 0.9761           | <b>0.9013 </b>  |
| Jaccard Index    | 0.9571           | 0.9534           | <b>0.8234    </b>  |

## Training Results

![278413728-920f8165-c671-4550-ab8a-2be16ff42d26](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/42dcce9d-fc04-4ca8-9058-93780bf929f1)


    
## Predictions

![278413412-e1d49df3-fe6e-4b74-99b0-27c6589d1c40](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/61d816fd-d0dc-4974-b840-b246be0eda8d)
![278413439-546c20b3-6379-41da-a492-1f634aa4faf0](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/3c1a85a3-949f-48c2-9742-6689078604e5)

    

## References
1. https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model
2. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1802.02611
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
