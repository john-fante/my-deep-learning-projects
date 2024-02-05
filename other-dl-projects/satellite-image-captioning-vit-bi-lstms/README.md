## Satellite Image Captioning (ViT, Bi-LSTMs)

(kaggle link -> https://www.kaggle.com/code/banddaniel/satellite-image-captioning-vit-bi-lstms)

I have used the following methods.

* I tried to implementation of distributed deep learning strategy,
* I used a pretrained ViT model for image feature extraction [1],
* Used <b>tf.data</b> and <b>Data Generator</b> for input pipeline,
* rectified and recreated functions in this notebook [2],

* <b><i> I tried a mirrored strategy of using 2 GPU at the same time in the training stage, but this gave rise to an infinite loop.</i></b>

## Prediction Pipeline

![download (4)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/7b4fca5f-1c5a-485c-9768-1dfae4ed7d75)


## Predictions

![download (5)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/be3a25c7-c5a9-45c2-a8b6-faeb303ca6f1)



## References
1. https://github.com/faustomorales/vit-keras
2. https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstms
