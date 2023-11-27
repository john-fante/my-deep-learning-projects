# Dental X-Rays Classification

I tried a custom CNN model and other pretrained models (Xception,ResNet50, MobileNet, EfficientNetB0, ResNet101). However, I haven't obtained sufficient results in respect of accuracy and F1 score. This dataset is unbalanced. For example, the Cavity class has only 22 samples in the test set. I tried the 'class weights' method, but there was no improvement. I used a basic data augmentation method.

I have used the following methods.

* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,
* I used several image processing methods for images <b>(Contrast Limited AHE (CLAHE)</b>[1])
* I used a pretrained <b>DenseNet201</b> architecture for the feature extraction stage,
* <b>elu</b> activation function during the classification stage,

## Image Processing Operation
![download (26)](https://github.com/john-fante/dental-xrays-classification/assets/50263592/0e6e65b6-4965-4aaa-ab14-2624f9b1bdab)

## Training Results
![__results___17_1](https://github.com/john-fante/dental-xrays-classification/assets/50263592/b9f4e481-cd38-49bb-8305-b732c0ca859d)


## Test Set Predictions
![download (28)](https://github.com/john-fante/dental-xrays-classification/assets/50263592/6cacd576-4819-4b88-af68-1a5051f05172)
![download (27)](https://github.com/john-fante/dental-xrays-classification/assets/50263592/165ba7af-6045-4e28-8298-52b2cd34b3e4)


