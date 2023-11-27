# Fungus Detection w/10 Kfold CV Custom ConvMixer (F1 : 0.85)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/fungus-detect-w-10cv-custom-convmixer-f1-0-85 </b>

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* Used <b>tf.data</b> for input pipeline,
* 10 Kfold cross-validation,
* I split the full data into train (6563 images), validation (729 images) and test (1822 images),
* Applying ensemble method to 10-fold test predictions


![280655958-e8eb88e7-2820-4068-9444-d625a1989a1f](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/50b1aa05-2bb8-40bb-ad84-4b1a9681edb7)


<i> ConvMixer Layer from the paper [1]</i>

## Test Set Predictions

![280651986-ddf5ee95-586e-4da6-9135-70789bb7f0c7](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/bd4eda47-f1a2-4143-9407-e5f63b292578)
![280652003-b7613204-ae1a-43a8-a96d-8a16a9997afd](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/e4d11350-6330-45b3-9251-ca0c01490cca)



## Test Results

![280652207-fa80db08-c4e2-4fad-abf2-2a71d5a0b853](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/7646bcc2-9a55-4881-ad5b-07f0a43ffefa)



## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
