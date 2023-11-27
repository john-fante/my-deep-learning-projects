# Jellyfish Classification (10KFold CV w/Custom ConvMixer) (F1:0.87)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/jellyfish-detect-10cv-custom-convmixer-f1-0-87 </b>

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* Used <b>tf.data</b> for input pipeline,
* 10 Kfold cross-validation,
* I split the full data into train (688 images), validation (77 images) and test (135 images),
* Applying ensemble method to 10-fold test predictions

![280655958-e8eb88e7-2820-4068-9444-d625a1989a1f](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/6f1c19db-9b7a-4b8e-8cb3-688efb5140a3)


<i> ConvMixer Layer from the paper [1]</i>

## Test Set Predictions

![280656338-3cccb1c8-00da-4ac7-ad1f-6cf74e8519ce](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/0b6e2c84-2a20-4fff-afa4-2f57b022eddf)
![280656371-35a55bf9-fc35-461c-92aa-79890c5dc731](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/488862e2-6f4c-473e-a148-724dbac56d04)



## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
