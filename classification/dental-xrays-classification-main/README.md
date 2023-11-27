# Dental X-Rays Classification

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/dental-x-rays-classification-test-f1-score-0-72 </b>

I tried a custom CNN model and other pretrained models (Xception,ResNet50, MobileNet, EfficientNetB0, ResNet101). However, I haven't obtained sufficient results in respect of accuracy and F1 score. This dataset is unbalanced. For example, the Cavity class has only 22 samples in the test set. I tried the 'class weights' method, but there was no improvement. I used a basic data augmentation method.

I have used the following methods.

* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,
* I used several image processing methods for images <b>(Contrast Limited AHE (CLAHE)</b>[1])
* I used a pretrained <b>DenseNet201</b> architecture for the feature extraction stage,
* <b>elu</b> activation function during the classification stage,

## Image Processing Operation

![278410447-0e6e65b6-4965-4aaa-ab14-2624f9b1bdab](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/486d0fee-15f9-47eb-a568-afd489bf660b)


## Training Results

![278410859-b9f4e481-cd38-49bb-8305-b732c0ca859d](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/6a56a929-d9c4-44ed-8336-74ea380d23da)


## Test Set Predictions

![278410465-6cacd576-4819-4b88-af68-1a5051f05172](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/e178f564-9bbb-4f76-81e1-5ea64d3b8793)
![278410491-165ba7af-6045-4e28-8298-52b2cd34b3e4](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/163870c5-13d7-4916-b63d-128b6a5747a4)



