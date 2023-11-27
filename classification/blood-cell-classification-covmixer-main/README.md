# Blood Cells Classification w/Custom ConvMixer (F1 Score: 0.98)

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,
* I split the full data into train (15425 images), validation (812 images) and test (855 images)


![download (28)](https://github.com/john-fante/blood-cell-classification-covmixer/assets/50263592/d24d7603-43f1-487e-bfae-1ef7bcf3ad21)

<i> ConvMixer Layer from the paper [1]</i>

## Test Set Predictions
![download (30)](https://github.com/john-fante/blood-cell-classification-covmixer/assets/50263592/83f1aa1f-d996-4b49-b54a-cb8b4b8529cf)
![download (29)](https://github.com/john-fante/blood-cell-classification-covmixer/assets/50263592/83daf45e-6ee4-441c-b103-42053ce4e80f)



## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
