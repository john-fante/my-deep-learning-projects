## (76 GB) 160 Polish Bird Sounds Classification

(kaggle link-> https://www.kaggle.com/code/banddaniel/76-gb-160-polish-bird-sounds-classification )


Firstly, I used TPU and pretrained models because it is a huge dataset. During training, total ram usage of TPU (8 chips) has reached nearly 310 GB out of 330 GB. Although I used TPU, the training phases took a long time. I tried a custom CNN model and other pretrained models (InceptionV3,  MobileNetV2	, NASNetLarge, EfficientNetV2B3). However, I haven't obtained sufficient results in respect of accuracy and F1 score. I uploaded some of my models and weights[1].

<br>
<b> Finally, I combined the 2 most accurate models and created an ensemble model. </b> 

* <span style="color:#e74c3c;">  Model 1 ResNet50 </span> 100x100 images, 8 Epochs, 64 Batches
* <span style="color:#e74c3c;">  Model 2 Xception </span> 100x100 images, 5 Epochs, 64 Batches
* Ensemble Model (25 % ResNet50, 75 % Xception)

<br>

I have used the following methods.

* The project took place using <b>Google TPU</b>,
* Used tf.data for input pipeline,
* Custom callback class that is used for saving the model and weights at the end of each epoch[2]


## Test Set Predictions
![download (35)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/5135ac20-4eb4-4685-90e5-afef34396fe1)
![download (36)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/edd1c509-fbdd-4cdf-81ad-eca8f956c57a)



## References
1. [Polish Bird Spectrograms My Models and Weights](https://www.kaggle.com/datasets/banddaniel/bird-sounds-h5)
2. My another custom callbacks for Tensorflow (https://github.com/john-fante/my-tensorflow-custom-callbacks)
