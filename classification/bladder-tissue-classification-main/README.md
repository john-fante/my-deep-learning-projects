# Bladder Tissue Classification w/ViT (F1 Score: 0.82)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/bladder-tissue-classification-w-vit-f1-scr-0-82 </b>

Firstly, I have used pretrained convolutional based models (a custom CNN, VGG19, Inception, ResNet101 etc.), but haven't obtained a good F1 Score in the test prediction. Then I tried a pretrained ViT model with preprocessed images I obtained good results. 

<br>
In addition, training this type of datasets can give rise to overfit easily due to the data leakage problem [1,2]. Our model shouldn't see the test set's samples. In this dataset, the file name format is 'case_002...'. I split all the same case files into train, validation and test datasets by hand (DataFrame splitting). For example, in all case_009 files only are used in the train dataset. 


I have used the following methods.

* I used a mirrored strategy (using 2 T4 GPU at the same time),
* Used <b>tf.data</b> for input pipeline,
* I used two image processing methods for images <span style="color:#e74c3c;"> <b>(Green Channel Conversion[3] , Histogram Equalization[4])</b> </span>
* I used a pretrained <b>ViT (Vision Transformer)</b> architecture [5],
* <b>gelu</b> activation function during the classification stage,


## Image Processing Operation

![280665833-e4c2027e-ff88-472b-bc53-a074dcefac91](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/1bf0da09-f5fa-4e64-9486-a8ef20388cf5)


## Test Results

![280666017-47cb47d4-cc85-4cef-a65e-d53578889a29](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/803e480c-9dd4-40f7-8513-669352dd79c0)


## Test Set Predictions

![280665793-ddb1256d-84a9-4253-ab66-4c21d39d8859](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/1fa20847-03f8-411d-93f7-3fcd5319de9c)


## References
1. https://en.wikipedia.org/wiki/Leakage_(machine_learning)
2. https://dataintegration.info/detect-multicollinearity-target-leakage-and-feature-correlation-with-amazon-sagemaker-data-wrangler
3. Rathod, Deepali & Manza, Ramesh & Rajput, Yogesh & Patwari, Manjiri & Saswade, Manoj & Deshpande, Neha. (2014). Localization of Optic Disc and Macula using Multilevel 2-D Wavelet Decomposition Based on Haar Wavelet Transform. International Journal of Engineering Research & Technology (IJERT)
4. https://en.wikipedia.org/wiki/Histogram_equalization
5. https://pypi.org/project/keras-vit/1.0.1/
