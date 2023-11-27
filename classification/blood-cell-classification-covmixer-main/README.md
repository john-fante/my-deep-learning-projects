# Blood Cells Classification w/Custom ConvMixer (F1 Score: 0.98)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/blood-cells-clf-w-custom-convmixer-f1-scr-0-98 </b>

I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,
* I split the full data into train (15425 images), validation (812 images) and test (855 images)


![278712650-d24d7603-43f1-487e-bfae-1ef7bcf3ad21](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/53dd3800-49ca-45f1-ade6-4966689ec3f2)



<i> ConvMixer Layer from the paper [1]</i>

## Test Set Predictions

![278712675-83f1aa1f-d996-4b49-b54a-cb8b4b8529cf](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/f2945992-4c2a-49e9-b1e5-391bc0288ed4)
![278712700-83daf45e-6ee4-441c-b103-42053ce4e80f](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/31f07608-6431-4947-9857-14df087514b5)



## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
