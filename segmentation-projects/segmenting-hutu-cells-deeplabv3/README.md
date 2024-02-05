## Segmenting HuTu Cells DeepLabv3+ (Test Dice: 0.93)

(kaggle link -> https://www.kaggle.com/code/banddaniel/segmenting-hutu-cells-deeplabv3-test-dice-0-93)


I have used the following methods.

* I have implemented DeepLabv3+ stemmed from this Keras example[1,2],
* I inverted the original masks <b><span style="color:#e74c3c;"></span></b> in the image preprocessing stage,
* <b>Dice coefficient</b> and <b>Jaccard index</b> implementation[3],
* I used <b><span style="color:#e74c3c;"> DeepLabv3+ implementation</span></b> with ResNet50 backbone,
* Used tf.data for input pipeline,
* <b>A Custom layer</b> for convolution operation,
* <b>A Custom layer</b> for Dilated Spatial Pyramid Pooling operation,
* Splitting dataset for training, validating and testing
    
    
<b> <span style="color:#2980b9;"> Note : In addition, training this type of datasets can give rise to overfit easily due to the data leakage problem [4]. Our model shouldn't see the test set's samples. In this dataset, the file name format is '0(control)_1_4(low)_040619_...'. I split all the same case files into train, validation and test datasets by hand (DataFrame splitting). For example, in all 0(control)_1_4(low)_040619_0h-72h files only are used in the validation dataset. </b> </span>


    
## Test Set Predictions

![download (12)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/056f3558-d5d5-4d3c-9e12-c5c4fcd79f77)


## References
1. https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model
2. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1802.02611
3. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
4. https://en.wikipedia.org/wiki/Leakage_(machine_learning)
