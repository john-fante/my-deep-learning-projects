# 30 Plants Detection w/Custom ConvMixer (F1 Scr: 0.77)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/30-plants-detect-w-custom-convmixer-f1-scr-0-77 </b>

First of all, I am very keen on trying new methods.
<br>

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,


![278712650-d24d7603-43f1-487e-bfae-1ef7bcf3ad21](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/281f8359-8dec-4a68-855b-2451551632b4)


<i> ConvMixer Layer from the paper [1]</i>


## Results

![280661603-9662d36a-ccd0-46ec-9d2e-39717dfccd77](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/e45bb023-2e8a-4fc8-b3ff-622734a158d9)


## Test Set Predictions

![280661409-21920f81-72f1-4bb3-8e52-c735015ea780](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/08668087-c90b-40d5-876e-35f32605a2bd)
![280661435-3a1096e6-1e72-498c-94b9-739395a0cca5](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/f2a6f5a7-23a5-4e1b-95c3-79e0cf19e3c6)



## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
