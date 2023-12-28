## Maize Classification w/Custom ResNet18 (AUC 0.928)

(kaggle link -> https://www.kaggle.com/code/banddaniel/maize-classification-w-custom-resnet18-auc-0-928)

I have used the following methods.

* I used a custom ResNet-18[1] architecture,
* <b>A custom layer</b> for residual block,
* Splitted train(3876 images) and test set (970 images),
* Used tf.data for input pipeline,

## Lightweight ResNet-18 Model

![download (40)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/1c6b71b9-6656-4d9b-8c3f-24e06d8966b6)


## Test Set Predictions
![download (41)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/0040817d-7297-4256-98f0-b8fcfad8d13e)
![download (39)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/e7308e84-8d6d-4701-9b8a-c33d1ebdfd28)



## References
1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1512.03385
