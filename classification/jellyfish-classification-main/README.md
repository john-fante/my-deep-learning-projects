# Jellyfish Classification (10KFold CV w/Custom ConvMixer) (F1:0.87)

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* Used <b>tf.data</b> for input pipeline,
* 10 Kfold cross-validation,
* I split the full data into train (688 images), validation (77 images) and test (135 images),
* Applying ensemble method to 10-fold test predictions

![download (35)](https://github.com/john-fante/jellyfish-classification/assets/50263592/e8eb88e7-2820-4068-9444-d625a1989a1f)

<i> ConvMixer Layer from the paper [1]</i>

## Test Set Predictions
![download (39)](https://github.com/john-fante/jellyfish-classification/assets/50263592/3cccb1c8-00da-4ac7-ad1f-6cf74e8519ce)
![download (38)](https://github.com/john-fante/jellyfish-classification/assets/50263592/35a55bf9-fc35-461c-92aa-79890c5dc731)


## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
