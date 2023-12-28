## Rice Classification w/Custom ResNet50

(kaggle link -> https://www.kaggle.com/code/banddaniel/rice-classification-w-custom-resnet50-acc-85)

I have used the following methods.

* I used a custom ResNet-50[1] architecture,
* The project took place using <b>Google TPU</b>,
* I used the validation set for testing
* <b>elu</b> activation function
* <b>A custom layer</b> for residual block,
* <span style="color:#e74c3c;"> <b>NOTE:</b> Of course, the accuracy metric is very high (if there is a data leakage between the train dataset and the test dataset, there is a problem named overfitting), but the loss metric continued decreasing and lastly the model has yielded reasonable results in respect of the confusion matrix. </span>


## Test Set Predictions
![download (38)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/23014808-f2da-4462-a58a-0bd36715d334)



## References
1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1512.03385
