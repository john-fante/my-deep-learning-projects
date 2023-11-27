# Fungus Detection w/10 Kfold CV Custom ConvMixer (F1 : 0.85)

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* Used <b>tf.data</b> for input pipeline,
* 10 Kfold cross-validation,
* I split the full data into train (6563 images), validation (729 images) and test (1822 images),
* Applying ensemble method to 10-fold test predictions


![download (35)](https://github.com/john-fante/fungus-detection/assets/50263592/e60fdaac-8fb1-4a08-a803-0a754f924149)

<i> ConvMixer Layer from the paper [1]</i>

## Test Set Predictions
![download (37)](https://github.com/john-fante/fungus-detection/assets/50263592/ddf5ee95-586e-4da6-9135-70789bb7f0c7)
![download (36)](https://github.com/john-fante/fungus-detection/assets/50263592/b7613204-ae1a-43a8-a96d-8a16a9997afd)

## Test Results
![__results___22_1](https://github.com/john-fante/fungus-detection/assets/50263592/fa80db08-c4e2-4fad-abf2-2a71d5a0b853)


## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
