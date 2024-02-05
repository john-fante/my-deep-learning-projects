## Road Segmentation w/DeepLabv3+ from Scratch

(kaggle link-> https://www.kaggle.com/code/banddaniel/road-segmentation-w-deeplabv3-from-scratch)

I have used the following methods.

* I have implemented DeepLabv3+ stemmed from this Keras example[1,2],
* I inverted the original masks <b><span style="color:#e74c3c;"></span></b> in the image preprocessing stage,
* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation[3],
* I used <b><span style="color:#e74c3c;"> DeepLabv3+ implementation</span></b> with ResNet50 backbone,
* Used tf.data for input pipeline,
* <b>A Custom layer</b> for convolution operation,
* <b>A Custom layer</b> for Dilated Spatial Pyramid Pooling operation,
* I don't use validation dataset
    
 
    
## Test Set Predictions

![download (11)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/dfed450b-59a1-405f-978a-2288bab58daa)



## References
1. https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model
2. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1802.02611
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
4. https://en.wikipedia.org/wiki/Leakage_(machine_learning)
