# 30 Plants Detection w/Custom ConvMixer (F1 Scr: 0.77)

First of all, I am very keen on trying new methods.
<br>

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,


![download (35)](https://github.com/john-fante/30-plants-classification/assets/50263592/12a25c47-d5a5-4291-ab40-60114d65994d)

<i> ConvMixer Layer from the paper [1]</i>


## Results
![__results___16_1](https://github.com/john-fante/30-plants-classification/assets/50263592/9662d36a-ccd0-46ec-9d2e-39717dfccd77)


## Test Set Predictions
![download (37)](https://github.com/john-fante/30-plants-classification/assets/50263592/21920f81-72f1-4bb3-8e52-c735015ea780)
![download (38)](https://github.com/john-fante/30-plants-classification/assets/50263592/3a1096e6-1e72-498c-94b9-739395a0cca5)


## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
